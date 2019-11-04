import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
from moz_sql_parser import parse
import time
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from networkx.algorithms import bipartite
import networkx as nx
import itertools
import hashlib
import psycopg2 as pg
import shelve
import pdb

# get rid of these
import getpass

ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
RANGE_PREDS = ["gt", "gte", "lt", "lte"]
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

'''
functions copied over from ryan's utils files
'''

def connected_subgraphs(g):
    for i in range(2, len(g)+1):
        for nodes_in_sg in itertools.combinations(g.nodes, i):
            sg = g.subgraph(nodes_in_sg)
            if nx.is_connected(sg):
                yield tuple(sorted(sg.nodes))

def generate_subset_graph(g):
    subset_graph = nx.DiGraph()
    for csg in connected_subgraphs(g):
        subset_graph.add_node(csg)

    # group by size
    max_subgraph_size = max(len(x) for x in subset_graph.nodes)
    subgraph_groups = [[] for _ in range(max_subgraph_size)]
    for node in subset_graph.nodes:
        subgraph_groups[len(node)-1].append(node)

    for g1, g2 in zip(subgraph_groups, subgraph_groups[1:]):
        for superset in g2:
            super_as_set = set(superset)
            for subset in g1:
                assert len(superset) == len(subset) + 1
                if set(subset) < super_as_set:
                    subset_graph.add_edge(superset, subset)
                    
    return subset_graph

def get_optimal_edges(sg):
    paths = {}
    orig_sg = sg
    sg = sg.copy()
    while len(sg.nodes) != 0:
        # first, find the root(s) of the subgraph at the highest level
        roots = {n for n,d in sg.in_degree() if d == 0}
        max_size_root = len(max(roots, key=lambda x: len(x)))
        roots = {r for r in roots if len(r) == max_size_root}
        
        # find everything within reach of 1
        reach_1 = set()
        for root in roots:
            reach_1.update(sg.neighbors(root))

        # build a bipartite graph and do the matching
        all_nodes = reach_1 | roots
        bipart_layer = sg.subgraph(all_nodes).to_undirected()
        assert(bipartite.is_bipartite(bipart_layer))
        matching = bipartite.hopcroft_karp_matching(bipart_layer, roots)
        matching = { k: v for k,v in matching.items() if k in roots}

        # sanity check -- every vertex should appear in exactly one path
        assert len(set(matching.values())) == len(matching)
        
        # find unmatched roots and add a path to $, indicating that
        # the path has terminated.
        for unmatched_root in roots - matching.keys():
            matching[unmatched_root] = "$"
        assert len(matching) == len(roots)

        # sanity check -- nothing was already in our paths
        for k, v in matching.items():
            assert k not in paths.keys()
            assert v not in paths.keys()                
            assert v == "$" or v not in paths.values()

        # sanity check -- all roots have an edge assigned
        for root in roots:
            assert root in matching.keys()
        
        paths.update(matching)

        # remove the old roots
        sg.remove_nodes_from(roots)
    return paths

def reconstruct_paths(edges):
    g = nx.Graph()
    for pair in edges.items():
        g.add_nodes_from(pair)

    for v1, v2 in edges.items():
        if v2 != "$":
            assert len(v1) > len(v2) and set(v1) > set(v2)
        g.add_edge(v1, v2)


    if "$" in g.nodes:
        g.remove_node("$")

    for node in g.nodes:
        assert g.degree(node) <= 2, f"{node} had degree of {g.degree(node)}"
        
    conn_comp = nx.algorithms.components.connected_components(g)
    paths = (sorted(x, key=len, reverse=True) for x in conn_comp)
    return paths

def greedy(subset_graph, plot=False):
    subset_graph = subset_graph.copy()

    while subset_graph:
        longest_path = nx.algorithms.dag.dag_longest_path(subset_graph)
        if plot:
            display(draw_graph(subset_graph, highlight_nodes=longest_path))
        subset_graph.remove_nodes_from(longest_path)
        yield longest_path

def path_to_join_order(path):
    remaining = set(path[0])
    for node in path[1:]:
        diff = remaining - set(node)
        yield diff
        remaining -= diff
    yield remaining

def order_to_from_clause(join_graph, join_order, alias_mapping):
    clauses = []
    for rels in join_order:
        if len(rels) > 1:
            # we should ask PG for an ordering here, since there's
            # no way to specify that the optimizer should control only these
            # bottom-level joins.
            sg = join_graph.subgraph(rels)
            sql = nx_graph_to_query(sg)
            con = pg.connect(user="imdb", host="localhost", database="imdb")
            pg_order = get_pg_join_order(sql, join_graph, con)
            assert not clauses
            clauses.append(pg_order)
            continue

        clause = f"{alias_mapping[rels[0]]} as {rels[0]}" 
        clauses.append(clause)

    return " CROSS JOIN ".join(clauses)

join_types = set(["Nested Loop", "Hash Join", "Merge Join"])

def extract_aliases(plan, jg=None):
    if "Alias" in plan:
        assert plan["Node Type"] == "Bitmap Heap Scan" or "Plans" not in plan
        if jg:
            alias = plan["Alias"]
            real_name = jg.nodes[alias]["real_name"]
            yield f"{real_name} as {alias}"
        else:
            yield plan["Alias"]

    if "Plans" not in plan:
        return

    for subplan in plan["Plans"]:
        yield from extract_aliases(subplan, jg=jg)

def analyze_plan(plan):
    if plan["Node Type"] in join_types:
        aliases = extract_aliases(plan)
        data = {"aliases": list(sorted(aliases))}
        if "Plan Rows" in plan:
            data["expected"] = plan["Plan Rows"]
        if "Actual Rows" in plan:
            data["actual"] = plan["Actual Rows"]

        yield data

    if "Plans" not in plan:
        return

    for subplan in plan["Plans"]:
        yield from analyze_plan(subplan)

'''
functions copied over from pari's util files
'''

def nx_graph_to_query(G, from_clause=None):
    froms = []
    conds = []
    for nd in G.nodes(data=True):
        node = nd[0]
        data = nd[1]
        if "real_name" in data:
            froms.append(ALIAS_FORMAT.format(TABLE=data["real_name"],
                                             ALIAS=node))
        else:
            froms.append(node)

        for pred in data["predicates"]:
            conds.append(pred)

    for edge in G.edges(data=True):
        conds.append(edge[2]['join_condition'])

    # preserve order for caching
    froms.sort()
    conds.sort()
    from_clause = " , ".join(froms) if from_clause is None else from_clause
    if len(conds) > 0:
        wheres = ' AND '.join(conds)
        from_clause += " WHERE " + wheres
    count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE=from_clause)
    return count_query

def extract_join_clause(query):
    '''
    FIXME: this can be optimized further / or made to handle more cases
    '''
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    start = time.time()
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    if where_clauses is None:
        return []
    join_clauses = []

    froms, aliases, table_names = extract_from_clause(query)
    if len(aliases) > 0:
        tables = [k for k in aliases]
    else:
        tables = table_names
    matches = find_all_clauses(tables, where_clauses)

    for match in matches:
        if "=" not in match or match.count("=") > 1:
            continue
        if "<=" in match or ">=" in match:
            continue

        match = match.replace(";", "")
        left, right = match.split("=")
        # ugh dumb hack
        if "." in right:
            # must be a join, so add it.
            join_clauses.append(left.strip() + " = " + right.strip())

    return join_clauses

def get_all_wheres(parsed_query):
    pred_vals = []
    if "where" not in parsed_query:
        pass
    elif "and" not in parsed_query["where"]:
        pred_vals = [parsed_query["where"]]
    else:
        pred_vals = parsed_query["where"]["and"]
    return pred_vals

# FIXME: get rid of this dependency
def extract_predicates(query):
    '''
    @ret:
        - column names with predicate conditions in WHERE.
        - predicate operator type (e.g., "in", "lte" etc.)
        - predicate value
    Note: join conditions don't count as predicate conditions.

    FIXME: temporary hack. For range queries, always returning key
    "lt", and vals for both the lower and upper bound
    '''
    def parse_column(pred, cur_pred_type):
        '''
        gets the name of the column, and whether column location is on the left
        (0) or right (1)
        '''
        for i, obj in enumerate(pred[cur_pred_type]):
            assert i <= 1
            if isinstance(obj, str) and "." in obj:
                # assert "." in obj
                column = obj
            elif isinstance(obj, dict):
                assert "literal" in obj
                val = obj["literal"]
                val_loc = i
            else:
                val = obj
                val_loc = i

        assert column is not None
        assert val is not None
        return column, val_loc, val

    def _parse_predicate(pred, pred_type):
        if pred_type == "eq":
            columns = pred[pred_type]
            if len(columns) <= 1:
                return None
            # FIXME: more robust handling?
            if "." in str(columns[1]):
                # should be a join, skip this.
                # Note: joins only happen in "eq" predicates
                return None
            predicate_types.append(pred_type)
            predicate_cols.append(columns[0])
            predicate_vals.append(columns[1])

        elif pred_type in RANGE_PREDS:
            vals = [None, None]
            col_name, val_loc, val = parse_column(pred, pred_type)
            vals[val_loc] = val

            # this loop may find no matching predicate for the other side, in
            # which case, we just leave the val as None
            for pred2 in pred_vals:
                pred2_type = list(pred2.keys())[0]
                if pred2_type in RANGE_PREDS:
                    col_name2, val_loc2, val2 = parse_column(pred2, pred2_type)
                    if col_name2 == col_name:
                        # assert val_loc2 != val_loc
                        if val_loc2 == val_loc:
                            # same predicate as pred
                            continue
                        vals[val_loc2] = val2
                        break

            predicate_types.append("lt")
            predicate_cols.append(col_name)
            if "g" in pred_type:
                # reverse vals, since left hand side now means upper bound
                vals.reverse()
            predicate_vals.append(vals)

        elif pred_type == "between":
            # we just treat it as a range query
            col = pred[pred_type][0]
            val1 = pred[pred_type][1]
            val2 = pred[pred_type][2]
            vals = [val1, val2]
            predicate_types.append("lt")
            predicate_cols.append(col)
            predicate_vals.append(vals)
        elif pred_type == "in" \
                or "like" in pred_type:
            # includes preds like, ilike, nlike etc.
            column = pred[pred_type][0]
            # what if column has been seen before? Will just be added again to
            # the list of predicates, which is the correct behaviour
            vals = pred[pred_type][1]
            if isinstance(vals, dict):
                vals = vals["literal"]
            if not isinstance(vals, list):
                vals = [vals]
            predicate_types.append(pred_type)
            predicate_cols.append(column)
            predicate_vals.append(vals)
        else:
            # TODO: need to support "OR" statements
            return None
            # assert False, "unsupported predicate type"

    start = time.time()
    predicate_cols = []
    predicate_types = []
    predicate_vals = []
    parsed_query = parse(query)
    pred_vals = get_all_wheres(parsed_query)

    # print("starting extract predicate cols!")
    for i, pred in enumerate(pred_vals):
        assert len(pred.keys()) == 1
        pred_type = list(pred.keys())[0]
        _parse_predicate(pred, pred_type)

    # print("extract predicate cols done!")
    # print("extract predicates took ", time.time() - start)
    return predicate_cols, predicate_types, predicate_vals

def extract_from_clause(query):
    '''
    Optimized version using sqlparse.
    Extracts the from statement, and the relevant joins when there are multiple
    tables.
    @ret: froms:
          froms: [alias1, alias2, ...] OR [table1, table2,...]
          aliases:{alias1: table1, alias2: table2} (OR [] if no aliases present)
          tables: [table1, table2, ...]
    '''
    def handle_table(identifier):
        table_name = identifier.get_real_name()
        alias = identifier.get_alias()
        tables.append(table_name)
        if alias is not None:
            from_clause = ALIAS_FORMAT.format(TABLE = table_name,
                                ALIAS = alias)
            froms.append(from_clause)
            aliases[alias] = table_name
        else:
            froms.append(table_name)

    start = time.time()
    froms = []
    # key: alias, val: table name
    aliases = {}
    # just table names
    tables = []

    start = time.time()
    parsed = sqlparse.parse(query)[0]
    # let us go over all the where clauses
    from_token = None
    from_seen = False
    for token in parsed.tokens:
        # print(type(token))
        # print(token)
        if from_seen:
            if isinstance(token, IdentifierList) or isinstance(token,
                    Identifier):
                from_token = token
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True

    assert from_token is not None
    if isinstance(from_token, IdentifierList):
        for identifier in from_token.get_identifiers():
            handle_table(identifier)
    elif isinstance(from_token, Identifier):
        handle_table(from_token)
    else:
        assert False

    return froms, aliases, tables

def find_next_match(tables, wheres, index):
    '''
    ignore everything till next
    '''
    match = ""
    _, token = wheres.token_next(index)
    if token is None:
        return None, None
    # FIXME: is this right?
    if token.is_keyword:
        index, token = wheres.token_next(index)

    tables_in_pred = find_all_tables_till_keyword(token)
    assert len(tables_in_pred) <= 2

    token_list = sqlparse.sql.TokenList(wheres)

    while True:
        index, token = token_list.token_next(index)
        if token is None:
            break
        # print("token.value: ", token.value)
        if token.value.upper() == "AND":
            break

        match += " " + token.value

        if (token.value.upper() == "BETWEEN"):
            # ugh ugliness
            index, a = token_list.token_next(index)
            index, AND = token_list.token_next(index)
            index, b = token_list.token_next(index)
            match += " " + a.value
            match += " " + AND.value
            match += " " + b.value
            # Note: important not to break here! Will break when we hit the
            # "AND" in the next iteration.

    # print("tables: ", tables)
    # print("match: ", match)
    # print("tables in pred: ", tables_in_pred)
    for table in tables_in_pred:
        if table not in tables:
            # print(tables)
            # print(table)
            # pdb.set_trace()
            # print("returning index, None")
            return index, None

    if len(tables_in_pred) == 0:
        return index, None

    return index, match

def find_all_clauses(tables, wheres):
    matched = []
    # print(tables)
    index = 0
    while True:
        index, match = find_next_match(tables, wheres, index)
        # print("got index, match: ", index)
        # print(match)
        if match is not None:
            matched.append(match)
        if index is None:
            break

    return matched

def find_all_tables_till_keyword(token):
    tables = []
    # print("fattk: ", token)
    index = 0
    while (True):
        if (type(token) == sqlparse.sql.Comparison):
            left = token.left
            right = token.right
            if (type(left) == sqlparse.sql.Identifier):
                tables.append(left.get_parent_name())
            if (type(right) == sqlparse.sql.Identifier):
                tables.append(right.get_parent_name())
            break
        elif (type(token) == sqlparse.sql.Identifier):
            tables.append(token.get_parent_name())
            break
        try:
            index, token = token.token_next(index)
            if ("Literal" in str(token.ttype)) or token.is_keyword:
                break
        except:
            break

    return tables

def execute_query(sql, user, db_host, port, pwd, db_name, pre_execs):
    '''
    @db_host: going to ignore it so default localhost is used.
    @pre_execs: options like set join_collapse_limit to 1 that are executed
    before the query.

    executes the given sql on the DB, and caches the results in a
    persistent store if it took longer than self.execution_cache_threshold.
    '''

    start = time.time()

    # FIXME: this needs consistent handling
    os_user = getpass.getuser()
    if os_user == "ubuntu":
        # for aws
        con = pg.connect(user=user, port=port,
                password=pwd, database=db_name)
    else:
        # for chunky
        con = pg.connect(user=user, host=db_host, port=port,
                password=pwd, database=db_name)

    cursor = con.cursor()

    for setup_sql in pre_execs:
        cursor.execute(setup_sql)

    try:
        cursor.execute(sql)
    except Exception as e:
        print(e)
        try:
            con.commit()
            cursor.close()
            con.close()
        finally:
            if not "timeout" in str(e):
                print("failed to execute for reason other than timeout")
                print(e)
            return None

    exp_output = cursor.fetchall()
    cursor.close()
    con.close()
    end = time.time()

    return exp_output

def deterministic_hash(string):
    return hashlib.sha1(str(string).encode("utf-8")).hexdigest()

def get_pg_join_order(sql, join_graph, con):
    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))

            if len(left) == 1 and len(right) == 1:
                return left[0] +  " CROSS JOIN " + right[0]

            if len(left) == 1:
                return left[0] + " CROSS JOIN (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                return "(" + __extract_jo(plan["Plans"][0]) + ") CROSS JOIN " + right[0]

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") CROSS JOIN ("
                    + __extract_jo(plan["Plans"][1]) + ")")
        
        return __extract_jo(plan["Plans"][0])
    
    cursor = con.cursor()

    cursor.execute(f"explain (format json) {sql}")
    exp_output = cursor.fetchall()
    cursor.close()
    con.close()

    return __extract_jo(exp_output[0][0][0]["Plan"])

def extract_join_graph(sql):
    '''
    @sql: string
    '''
    froms,aliases,tables = extract_from_clause(sql)
    joins = extract_join_clause(sql)
    join_graph = nx.Graph()

    for j in joins:
        j1 = j.split("=")[0]
        j2 = j.split("=")[1]
        t1 = j1[0:j1.find(".")].strip()
        t2 = j2[0:j2.find(".")].strip()
        try:
            assert t1 in tables or t1 in aliases
            assert t2 in tables or t2 in aliases
        except:
            print(t1, t2)
            print(tables)
            print(joins)
            print("table not in tables!")
            pdb.set_trace()

        join_graph.add_edge(t1, t2)
        join_graph[t1][t2]["join_condition"] = j
        if t1 in aliases:
            table1 = aliases[t1]
            table2 = aliases[t2]

            join_graph.nodes()[t1]["real_name"] = table1
            join_graph.nodes()[t2]["real_name"] = table2

    parsed = sqlparse.parse(sql)[0]
    # let us go over all the where clauses
    where_clauses = None
    for token in parsed.tokens:
        if (type(token) == sqlparse.sql.Where):
            where_clauses = token
    assert where_clauses is not None

    for t1 in join_graph.nodes():
        tables = [t1]
        matches = find_all_clauses(tables, where_clauses)
        join_graph.nodes()[t1]["predicates"] = matches

    return join_graph

def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    # if "Scan" in v:
                        # print(v)
                        # pdb.set_trace()
                    # if "Join" in v:
                        # print(obj)
                        # pdb.set_trace()
                    arr.append(v)

        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results
