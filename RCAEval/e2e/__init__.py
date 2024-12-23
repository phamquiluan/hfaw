import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)
from RCAEval.utility import is_py310


def rca(func):
    """RCA Wrapper to tolerate the case when the RCA algorithm fails."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            from RCAEval.io.time_series import preprocess
            data = preprocess(data=args[0], dataset=kwargs.get("dataset"), dk_select_useful=False)
            dummy = data.columns.to_list()
            return {"adj": [], "node_names": dummy, "ranks": dummy}
    return wrapper

if is_py310():
    try:
        from .causalai import causalai
    except Exception as e:
        pass
    from .baro import baro, mmbaro, mmnsigma
    from .causalrca import causalrca
    from .circa import circa
    from .cloudranger import cloudranger
    from .fci_pagerank import fci_pagerank
    from .ges_pagerank import ges_pagerank
    from .granger_pagerank import granger_pagerank
    from .lingam_pagerank import lingam_pagerank, micro_diag
    from .microcause import microcause
    from .microrank import microrank
    from .easyrca import easyrca
    from .pc_pagerank import cmlp_pagerank, ntlr_pagerank, pc_pagerank
    from .pc_randomwalk import (
        fci_randomwalk,
        granger_randomwalk,
        lingam_randomwalk,
        ntlr_randomwalk,
        pc_randomwalk,
    )
    from .run import run
    from .mscred import mscred
    from .tracerca import tracerca
else:
    from .rcd import rcd
    from .mmrcd import mmrcd


def dummy(data, inject_time=None, dataset=None, *args, **kwargs):
    """
    data: pd.DataFrame

    Return:
        adj: np.ndarray, adjacency matrix
        root_causes:
    """
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    cols = data.columns.to_list()

    # dummy graph discovery
    adj = np.zeros((len(cols), len(cols)))

    # random sort the cols
    root_causes = np.random.choice(cols, size=len(cols), replace=False).tolist()
    # return adj, root_causes
    return {"adj": adj, "ranks": root_causes}


def nsigma(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = StandardScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }





def e_diagnosis(
    data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs
):
    from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis

    alpha = float(os.getenv("E_ALPHA", 0.01))
    # print(f"=========== E alpha: {alpha} ===========")
    model = EpsilonDiagnosis(config=EpsilonDiagnosis.config_class(alpha=alpha))

    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

        print(f"{len(normal_df)=} {len(anomal_df)=}")

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
    min_length = min(normal_df.shape[0], anomal_df.shape[0])
    normal_df = normal_df.tail(min_length)
    anomal_df = anomal_df.head(min_length)

    model.train(normal_df)
    results = model.find_root_causes(anomal_df)
    ranks = results.to_dict()["root_cause_nodes"]

    # ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


def ht(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    from pyrca.analyzers.ht import HT, HTConfig
    from pyrca.graphs.causal.fges import FGES, FGESConfig
    from pyrca.graphs.causal.pc import PC

    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

        print(f"{len(normal_df)=} {len(anomal_df)=}")

    # preprocess data
    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
    min_length = min(normal_df.shape[0], anomal_df.shape[0])
    normal_df = normal_df.tail(min_length)
    anomal_df = anomal_df.head(min_length)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)
    data.bfill(inplace=True)
    data.ffill(inplace=True)
    data.fillna(0, inplace=True)

    fges_penalty = os.getenv("FGES_PENALTY", None)
    if fges_penalty is not None:
        # learn graph by fges
        fges_config = FGESConfig()
        fges_config.run_pdag2dag = False
        fges_config.penalty_discount = int(fges_penalty)
        print(f"=========== FGES penalty: {fges_config.penalty_discount} ===========")
        model = FGES(fges_config)
        graph_df = model.train(data)
    else:
        # learn graph by pc
        fges_config = FGESConfig()
        pc_config = PC.config_class()
        pc_config.run_pdag2dag = False
        pc_config.alpha = float(os.getenv("PC_ALPHA", 0.01))
        print(f"=========== PC alpha: {pc_config.alpha} ===========")
        model = PC(pc_config)
        graph_df = model.train(data)

    # rca
    model = HT(HTConfig(graph=graph_df))
    model.train(normal_df)
    results = model.find_root_causes(anomal_df)
    ranks = results.to_dict()["root_cause_nodes"]

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    print(ranks)

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }




