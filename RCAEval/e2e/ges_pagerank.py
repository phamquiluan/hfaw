from RCAEval.graph_construction.ges import ges
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.graph_heads.random_walk import random_walk
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca

def ges_pagerank(data, inject_time=None, dataset=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    record = ges(data)
    G = record["G"]
    ranks = page_rank(G.graph, node_names=data.columns.to_list())
    return {
        "adj": G.graph,
        "node_names": data.columns.to_list(),
        "ranks": ranks,
    }