import psycopg2 as pg
import getpass
from .utils import *

import pdb

PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''
PG_HINT_JOIN_TMP = "{JOIN_TYPE} ({TABLES}) "
PG_HINT_CARD_TMP = "Rows ({TABLES} #{CARD}) "
PG_HINT_SCAN_TMP = "{SCAN_TYPE}({TABLE}) "
PG_HINT_LEADING_TMP = "Leading ({JOIN_ORDER})"
PG_HINT_JOINS = {}
PG_HINT_JOINS["Nested Loop"] = "NestLoop"
PG_HINT_JOINS["Hash Join"] = "HashJoin"
PG_HINT_JOINS["Merge Join"] = "MergeJoin"

PG_HINT_SCANS = {}
PG_HINT_SCANS["Seq Scan"] = "SeqScan"
PG_HINT_SCANS["Index Scan"] = "IndexScan"
PG_HINT_SCANS["Index Only Scan"] = "IndexOnlyScan"
PG_HINT_SCANS["Bitmap Heap Scan"] = "BitmapScan"
PG_HINT_SCANS["Tid Scan"] = "TidScan"

MAX_JOINS = 16

def _get_cost(sql, cur):
    assert "explain" in sql
    # cur = con.cursor()
    cur.execute(sql)
    explain = cur.fetchall()
    all_costs = extract_values(explain[0][0][0], "Total Cost")
    mcost = max(all_costs)
    # cur.close()
    # cost = all_costs[-1]
    # pdb.set_trace()
    cost = explain[0][0][0]["Plan"]["Total Cost"]
    # if cost != mcost:
        # print(cost, mcost)
        # pdb.set_trace()
    return cost, explain

def _gen_pg_hint_cards(cards):
    '''
    '''
    card_str = ""
    for aliases, card in cards.items():
        card_line = PG_HINT_CARD_TMP.format(TABLES = aliases,
                                            CARD = card)
        card_str += card_line
    return card_str

def _gen_pg_hint_join(join_ops):
    '''
    '''
    join_str = ""
    for tables, join_op in join_ops.items():
        join_line = PG_HINT_JOIN_TMP.format(TABLES = tables,
                                            JOIN_TYPE = PG_HINT_JOINS[join_op])
        join_str += join_line
    return join_str

def _gen_pg_hint_scan(scan_ops):
    '''
    '''
    scan_str = ""
    for alias, scan_op in scan_ops.items():
        scan_line = PG_HINT_SCAN_TMP.format(TABLE = alias,
                                            SCAN_TYPE = PG_HINT_SCANS[scan_op])
        scan_str += scan_line
    return scan_str

def get_leading_hint(join_graph, explain):
    '''
    Ryan's implementation.
    '''
    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))

            if len(left) == 1 and len(right) == 1:
                left_alias = left[0][left[0].lower().find(" as ")+4:]
                right_alias = right[0][right[0].lower().find(" as ")+4:]
                return left_alias +  " " + right_alias

            if len(left) == 1:
                left_alias = left[0][left[0].lower().find(" as ")+4:]
                return left_alias + " (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                right_alias = right[0][right[0].lower().find(" as ")+4:]
                return "(" + __extract_jo(plan["Plans"][0]) + ") " + right_alias

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    jo = __extract_jo(explain[0][0][0]["Plan"])
    jo = "(" + jo + ")"
    return PG_HINT_LEADING_TMP.format(JOIN_ORDER = jo)

def get_pg_join_order(join_graph, explain):
    '''
    '''
    physical_join_ops = {}
    scan_ops = {}
    def __update_scan(plan):
        node_types = extract_values(plan, "Node Type")
        alias = extract_values(plan, "Alias")[0]
        for nt in node_types:
            if "Scan" in nt:
                scan_type = nt
                break
        scan_ops[alias] = nt

    def __extract_jo(plan):
        if plan["Node Type"] in join_types:
            left = list(extract_aliases(plan["Plans"][0], jg=join_graph))
            right = list(extract_aliases(plan["Plans"][1], jg=join_graph))
            all_froms = left + right
            all_nodes = []
            for from_clause in all_froms:
                from_alias = from_clause[from_clause.find(" as ")+4:]
                if "_info" in from_alias:
                    print(from_alias)
                    pdb.set_trace()
                all_nodes.append(from_alias)
            all_nodes.sort()
            all_nodes = " ".join(all_nodes)
            physical_join_ops[all_nodes] = plan["Node Type"]

            if len(left) == 1 and len(right) == 1:
                __update_scan(plan["Plans"][0])
                __update_scan(plan["Plans"][1])
                return left[0] +  " CROSS JOIN " + right[0]

            if len(left) == 1:
                __update_scan(plan["Plans"][0])
                return left[0] + " CROSS JOIN (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                __update_scan(plan["Plans"][1])
                return "(" + __extract_jo(plan["Plans"][0]) + ") CROSS JOIN " + right[0]

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") CROSS JOIN ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    return __extract_jo(explain[0][0][0]["Plan"]), physical_join_ops, scan_ops

def _get_modified_sql(sql, cardinalities, join_ops,
        leading_hint, scan_ops):
    '''
    @cardinalities: dict
    @join_ops: dict

    @ret: sql, augmented with appropriate comments.
    '''
    if "explain" not in sql:
        sql = " explain (format json) " + sql

    comment_str = ""
    if cardinalities is not None:
        card_str = _gen_pg_hint_cards(cardinalities)
        # gen appropriate sql with comments etc.
        comment_str += card_str

    if join_ops is not None:
        join_str = _gen_pg_hint_join(join_ops)
        comment_str += join_str + " "
    if leading_hint is not None:
        comment_str += leading_hint + " "
    if scan_ops is not None:
        scan_str = _gen_pg_hint_scan(scan_ops)
        comment_str += scan_str + " "

    pg_hint_str = PG_HINT_CMNT_TMP.format(COMMENT=comment_str)
    sql = pg_hint_str + sql
    return sql

def get_cardinalities_join_cost(query, est_cardinalities, true_cardinalities,
        join_graph, use_indexes):
    os_user = getpass.getuser()
    if os_user == "ubuntu":
        # FIXME: find single clean way to do this stuff
        try:
            con = pg.connect(port=5432,dbname="imdb",
                    user=os_user,password="")
        except:
            con = pg.connect(port=5432,dbname="imdb",
                    user=os_user,password="",host="localhost")
    else:
        con = pg.connect(host="localhost",port=5432,dbname="imdb",
                user="pari",password="")

    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))
    if not use_indexes:
        cursor.execute("SET enable_indexscan = off")
        cursor.execute("SET enable_indexonlyscan = off")
    else:
        cursor.execute("SET enable_indexscan = on")
        cursor.execute("SET enable_indexonlyscan = on")

    est_card_sql = _get_modified_sql(query, est_cardinalities, None,
            None, None)
    cursor.execute(est_card_sql)
    explain = cursor.fetchall()
    est_join_order_sql, est_join_ops, scan_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)
    assert "info" not in leading_hint

    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)

    # add the join ops etc. information
    cost_sql = _get_modified_sql(est_opt_sql, true_cardinalities,
            est_join_ops, leading_hint, scan_ops)

    exec_sql = _get_modified_sql(est_opt_sql, est_cardinalities,
            est_join_ops, leading_hint, scan_ops)

    est_cost, est_explain = _get_cost(cost_sql, cursor)
    debug_leading = get_leading_hint(join_graph, est_explain)

    if debug_leading != leading_hint:
        # print(est_opt_sql)
        print("actual order:\n ", debug_leading)
        print("wanted order:\n ", leading_hint)
        # pdb.set_trace()

    cursor.close()
    con.close()
    return exec_sql, est_cost, est_explain

def compute_join_order_loss_pg_single(query, true_cardinalities,
        est_cardinalities, opt_cost, opt_explain, opt_sql,
        use_indexes):
    '''
    @query: str
    @true_cardinalities:
        key:
            sort([table_1 / alias_1, ..., table_n / alias_n])
        val:
            float
    @est_cardinalities:
        key:
            sort([table_1 / alias_1, ..., table_n / alias_n])
        val:
            float

    '''
    # set est cardinalities
    # FIXME:
    if "mii1.info " in query:
        query = query.replace("mii1.info ", "mii1.info::float")
    if "mii2.info " in query:
        query = query.replace("mii2.info ", "mii2.info::float")
    if "mii1.info)" in query:
        query = query.replace("mii1.info)", "mii1.info::float)")
    if "mii2.info)" in query:
        query = query.replace("mii2.info)", "mii2.info::float)")

    # FIXME: we should not need join graph for all these helper methods
    join_graph = extract_join_graph(query)
    est_card_sql, est_cost, est_explain = get_cardinalities_join_cost(query,
            est_cardinalities, true_cardinalities, join_graph,
            use_indexes)
    if opt_cost is None:
        opt_sql, opt_cost, opt_explain = get_cardinalities_join_cost(query,
                true_cardinalities, true_cardinalities, join_graph,
                use_indexes)

    # adds the est cardinalities as a comment to the modified sql

    # FIXME: temporary
    if est_cost < opt_cost:
        # print(est_cost, opt_cost, opt_cost - est_cost)
        # pdb.set_trace()
        est_cost = opt_cost

    return est_cost, opt_cost, est_explain, opt_explain, est_card_sql, opt_sql
