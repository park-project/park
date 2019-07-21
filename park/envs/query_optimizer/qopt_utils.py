import psutil
import hashlib
import networkx as nx
from networkx.drawing.nx_agraph import write_dot,graphviz_layout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import math
import pdb

def plot_join_order(info, pdf, ep=0, title_suffix="", single_plot=True,
        python_alg_name="RL", est_cards=None, true_cards=None):
    '''
    @pdf: opened pdf file to which plots will be appended.
    '''
    def get_node_name(tables):
        name = ""
        if len(tables) > 1:
            name = str(deterministic_hash(str(tables)))[0:3]
            join_nodes.append(name)
        else:
            name = tables[0]
            # shorten it
            name = "".join([n[0] for n in name.split("_")])
            if name in base_table_nodes:
                name = name + "2"
            base_table_nodes.append(name)
        return name

    def format_ints(num):
        # returns the number formatted to closest 1000 + K
        return str(round(num, -3)).replace("000","") + "K"

    def plot_graph():
        NODE_SIZE = 300
        plt.title(title)
        pos = graphviz_layout(G, prog='dot')
        # first draw just the base tables
        nx.draw_networkx_nodes(G, pos,
                   nodelist=base_table_nodes,
                   node_color='b',
                   node_size=NODE_SIZE,
                   alpha=0.2)

        nx.draw_networkx_nodes(G, pos,
                   nodelist=join_nodes,
                   node_color='r',
                   node_size=NODE_SIZE,
                   alpha=0.2)

        if est_cards is not None and true_cards is not None:
            est_labels = {}
            est_card_pos = {}
            for k, v in pos.items():
                est_card_pos[k] = (v[0], v[1]-25)
                try:
                    est_labels[k] = format_ints(G.nodes[k]["est_card"])
                except:
                    est_labels[k] = -1
                    continue

            nx.draw_networkx_labels(G, est_card_pos, est_labels,
                    font_size=8, font_color='r')

            true_labels = {}
            true_label_pos = {}
            for k, v in pos.items():
                true_label_pos[k] = (v[0], v[1]+25)
                try:
                    true_labels[k] = format_ints(G.nodes[k]["true_card"])
                except:
                    true_labels[k] = -1

            nx.draw_networkx_labels(G, true_label_pos, true_labels,
                    font_size=8, font_color='g')

            # draw the within node labels
            node_labels = {}
            for n in G.nodes():
                if len(G.nodes[n]["tables"]) == 1:
                    node_labels[n] = n
                else:
                    try:
                        node_labels[n] = format_ints(G.nodes[n]["join_cost"])
                    except:
                        node_labels[n] = -1

            nx.draw_networkx_labels(G, pos, node_labels, font_size=8)

        else:
            node_labels = {}
            for n in G.nodes():
                if len(G.nodes[n]["tables"]) == 1:
                    node_labels[n] = n
                else:
                    node_labels[n] = format_ints(G.nodes[n]["join_cost"])

            nx.draw_networkx_labels(G, pos, node_labels, font_size=8)

        # if est_cards is not None:
            # est_labels = {}
            # est_card_pos = {}
            # for k, v in pos.items():
                # est_card_pos[k] = (v[0]+25, v[1])
                # est_labels[k] = format_ints(G.nodes[k]["est_card"])
            # nx.draw_networkx_labels(G, est_card_pos, est_labels,
                    # font_size=8, font_color='r')

        # if true_cards is not None:
            # true_labels = {}
            # true_label_pos = {}
            # for k, v in pos.items():
                # true_label_pos[k] = (v[0]-20, v[1])
                # true_labels[k] = format_ints(G.nodes[k]["true_card"])
            # nx.draw_networkx_labels(G, true_label_pos, true_labels,
                    # font_size=8, font_color='g')

        nx.draw_networkx_edges(G,pos,width=1.0,
                alpha=0.5,with_labels=False)
        plt.tight_layout()

    viz_title_tmp = "query: {query}, {alg} "
    query_name = os.path.basename(info["queryName"])
    # get relative cost baseline
    min_cost = min([v for _,v in info["costs"].items()])
    if not single_plot:
        plot_idx = 1
    for alg, jo in info["joinOrders"].items():
        G = nx.DiGraph()
        base_table_nodes = []
        join_nodes = []
        # TODO: node_labels should be set later, for now, just set all the
        # properties of the graph.
        # node_labels = {}
        for edge_idx, edge in enumerate(jo["joinEdges"]):
            assert len(edge) == 2
            node0 = get_node_name(edge[0])
            node1 = get_node_name(edge[1])
            # if len(edge[0]) == 1:
                # node_labels[node0] = node0

            G.add_edge(node0, node1)

            # add other parameters on the nodes
            G.nodes[node0]["tables"] = edge[0]
            G.nodes[node1]["tables"] = edge[1]
            # if len(edge[1]) > 1:
                # try:
                    # join_cost = int(jo["joinCosts"][str(edge[1]).replace("'","")])
                # except:
                    # join_cost = -1
                # node_labels[node1] = join_cost

            if not "joinCosts" in jo:
                # print(edge)
                # print(jo.keys())
                # pdb.set_trace()
                continue
            if true_cards is not None:
                G.nodes[node0]["true_card"] = true_cards[" " + " ".join(edge[0])]
                G.nodes[node1]["true_card"] = true_cards[" " + " ".join(edge[1])]

            if est_cards is not None:
                G.nodes[node0]["est_card"] = est_cards[" " + " ".join(edge[0])]
                G.nodes[node1]["est_card"] = est_cards[" " + " ".join(edge[1])]

            if len(edge[0]) > 1:
                G.nodes[node0]["join_cost"] = int(jo["joinCosts"][str(edge[0]).replace("'","")])

            if len(edge[1]) > 1:
                G.nodes[node1]["join_cost"] = int(jo["joinCosts"][str(edge[1]).replace("'","")])

        min_cost = max(min_cost, 0.0001)
        rel_cost = round(info["costs"][alg] / float(min_cost), 3)
        costK = str(round(info["costs"][alg]/1000)) + "K"
        if alg == "RL":
            alg = python_alg_name

        if single_plot:
            title = viz_title_tmp.format(query=query_name,
                    alg=alg) + title_suffix
        else:
            if plot_idx == 1:
                title = viz_title_tmp.format(query=query_name,
                        alg=alg) + title_suffix
            else:
                title = " " + alg
            title += " " + costK
            title += "(" + str(rel_cost) + ")"

        if single_plot:
            plot_graph()
            pdf.savefig()
            plt.close()
        else:
            plt.subplot(1, 2, plot_idx)
            plot_graph()
            plot_idx += 1

    if not single_plot:
        pdf.savefig()
        plt.close()

def find_available_port(orig_port):
    conns = psutil.net_connections()
    ports = [c.laddr.port for c in conns]
    new_port = orig_port

    while new_port in ports:
        new_port += 1
    return new_port

def deterministic_hash(string):
    return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)

