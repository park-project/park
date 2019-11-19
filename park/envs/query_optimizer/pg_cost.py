import psycopg2 as pg
import getpass
from .utils import *

import pdb

PG_HINT_CMNT_TMP = '''/*+ {COMMENT} */'''
PG_HINT_JOIN_TMP = "{JOIN_TYPE} ({TABLES})"
PG_HINT_CARD_TMP = "Rows ({TABLES} #{CARD})"
PG_HINT_LEADING_TMP = "Leading ({JOIN_ORDER})"
PG_HINT_JOINS = {}
PG_HINT_JOINS["Nested Loop"] = "NestLoop"
PG_HINT_JOINS["Hash Join"] = "HashJoin"
PG_HINT_JOINS["Merge Join"] = "MergeJoin"
MAX_JOINS = 16

def _get_cost(sql, cur):
    assert "explain" in sql
    # cur = con.cursor()
    cur.execute(sql)
    explain = cur.fetchall()
    all_costs = extract_values(explain[0][0][0], "Total Cost")
    cost = max(all_costs)
    # cur.close()
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
    Ryan's implementation.
    '''
    physical_join_ops = {}
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
                return left[0] +  " CROSS JOIN " + right[0]

            if len(left) == 1:
                return left[0] + " CROSS JOIN (" + __extract_jo(plan["Plans"][1]) + ")"

            if len(right) == 1:
                return "(" + __extract_jo(plan["Plans"][0]) + ") CROSS JOIN " + right[0]

            return ("(" + __extract_jo(plan["Plans"][0])
                    + ") CROSS JOIN ("
                    + __extract_jo(plan["Plans"][1]) + ")")

        return __extract_jo(plan["Plans"][0])

    return __extract_jo(explain[0][0][0]["Plan"]), physical_join_ops

def _get_modified_sql(sql, cardinalities, join_ops,
        leading_hint):
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
        # comment_str += join_str + " \n "
    if leading_hint is not None:
        # comment_str += leading_hint + "\n"
        comment_str += leading_hint + " "

    pg_hint_str = PG_HINT_CMNT_TMP.format(COMMENT=comment_str)
    # print(pg_hint_str)
    # pdb.set_trace()
    # sql = pg_hint_str + "\n" + sql
    sql = pg_hint_str + sql
    return sql

def compute_join_order_loss_pg_single(query, true_cardinalities,
        est_cardinalities):
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
    os_user = getpass.getuser()
    if os_user == "ubuntu":
        con = pg.connect(port=5432,dbname="imdb",
                user=os_user,password="")
    else:
        con = pg.connect(host="localhost",port=5432,dbname="imdb",
                user="pari",password="")
    # adds the est cardinalities as a comment to the modified sql
    est_card_sql = _get_modified_sql(query, est_cardinalities, None,
            None)

    # find join order
    cursor = con.cursor()
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute("SET geqo_threshold = {}".format(MAX_JOINS))
    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))

    cursor.execute(est_card_sql)
    explain = cursor.fetchall()
    est_join_order_sql, est_join_ops = get_pg_join_order(join_graph,
            explain)
    leading_hint = get_leading_hint(join_graph, explain)
    assert "info" not in leading_hint

    est_opt_sql = nx_graph_to_query(join_graph,
            from_clause=est_join_order_sql)

    # add the join ops etc. information
    est_opt_sql = _get_modified_sql(est_opt_sql, true_cardinalities,
            est_join_ops, leading_hint)

    est_cost, est_explain = _get_cost(est_opt_sql, cursor)
    debug_leading = get_leading_hint(join_graph, est_explain)

    if debug_leading != leading_hint:
        # print(est_opt_sql)
        print("actual order:\n ", debug_leading)
        print("wanted order:\n ", leading_hint)
        pdb.set_trace()

    # this would not use cross join syntax, so should work fine with
    # join_collapse_limit = 1 as well.
    opt_sql = _get_modified_sql(query, true_cardinalities, None, None)

    cursor.execute("SET join_collapse_limit = {}".format(MAX_JOINS))
    cursor.execute("SET from_collapse_limit = {}".format(MAX_JOINS))
    opt_cost, opt_explain = _get_cost(opt_sql, cursor)

    # FIXME: temporary
    if est_cost < opt_cost:
        # print(est_cost, opt_cost, opt_cost - est_cost)
        # pdb.set_trace()
        est_cost = opt_cost

    cursor.close()
    con.close()
    return est_cost, opt_cost, est_explain, opt_explain
