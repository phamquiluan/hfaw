import os
import sys
import csv
import json
import math
import time
import datetime
import codecs
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

from functools import lru_cache, reduce
from typing import FrozenSet, List

# add frozenset 

from tqdm import tqdm
import numpy as np
from dateutil.parser import parse
import pandas as pd



def get_operation_slo(span_df):
    """ Calculate the mean of duration and variance of each operation
    :arg
        span_df: span data with operations and duration columns
    :return
        operation dict of the mean of and variance 
        {
            # operation: {mean, variance}
            "Currencyservice_Convert": [600, 3]}
        }   
    """
    operation_slo = {}
    for op in span_df["operation"].dropna().unique():
        #get mean and std of Duration column of the corresponding operation
        mean = round(span_df[span_df["operation"] == op]["duration"].mean() / 1_000, 2)
        std = round(span_df[span_df["operation"] == op]["duration"].std() / 1_000, 2)

        # operation_slo[op] = [mean, std]
        operation_slo[op] = { "mean": mean, "std": std }

    # print(json.dumps(operation_slo, sort_keys=True, indent=2))
    return operation_slo


def main():
    ALPHA=5
    T=10
    K=2

    metric_df = pd.read_csv("./data/mm-ob/checkoutservice_cpu/1/simple_metrics.csv")
    log_df = pd.read_csv("./data/mm-ob/checkoutservice_cpu/1/logs.csv")
    logts_df = pd.read_csv("./data/mm-ob/checkoutservice_cpu/1/logts.csv")
    span_df = pd.read_csv("./data/mm-ob/checkoutservice_cpu/1/traces.csv")
    span_df["methodName"] = span_df["methodName"].fillna(span_df["operationName"])
    span_df["operation"] = span_df["serviceName"] + "_" + span_df["methodName"]

    with open("./data/mm-ob/checkoutservice_cpu/1/inject_time.txt") as f:
        inject_time = int(f.readline())
        # inject_time = int(f.readline()) * 1_000_000
    
    # for metrics
    normal_metric_df = metric_df[metric_df["time"] < inject_time]
    metric_slo = {m: [normal_metric_df[m].mean(), normal_metric_df[m].std()] for m in normal_metric_df.columns if "time" not in m}
    metric_q = {m: 0 for m in normal_metric_df.columns if "time" not in m}
    
    anomal_metric_df = metric_df[metric_df["time"] >= inject_time]

    # fill metric_q by number of number in anomal_metric_df larger than 3 sigma of mean
    for m in metric_q.keys():
        metric_q[m] = len(anomal_metric_df[anomal_metric_df[m] >= metric_slo[m][0] + 3 * metric_slo[m][1]]) 

    # print(json.dumps(metric_q, indent=2, sort_keys=True))

    service_dict = {k.split("_")[0]: 0 for k in metric_q.keys()}
    
    # add score for each service using metric
    for k, v in metric_q.items():
        s, m = k.split("_")
        service_dict[s] += ALPHA * v


    # logs
    inject_time_log = inject_time * 1_000_000  # convert from seconds to microseconds for logs
    log_q = {c: 0 for c in log_df["container_name"].unique()}
    
    for c in log_q.keys():
        # count log contains `err` in the whole df 
        log_q[c] = len(log_df[(log_df["container_name"] == c) & (log_df["message"].str.contains("error|fail"))])
    
    # print(json.dumps(log_q, indent=2, sort_keys=True))
    for k, v in log_q.items():
        service_dict[k] += ALPHA * v

    # traces
    normal_span_df  = span_df[span_df["startTime"] + span_df["duration"] < inject_time_log]
    anomal_span_df  = span_df[span_df["startTime"] + span_df["duration"] >= inject_time_log]
 
    normal_slo = get_operation_slo(normal_span_df)

    anomal_span_df["mean"] = anomal_span_df["operation"].apply(lambda op: normal_slo[op]["mean"])
    anomal_span_df["std"] = anomal_span_df["operation"].apply(lambda op: normal_slo[op]["std"])
    anomal_span_df["abnormal"] = anomal_span_df["duration"] / 1_000 >= anomal_span_df["mean"] + 3 * anomal_span_df["std"]


    # q = deepcopy(log_q)
    trace_q = {k: 0 for k in anomal_span_df.serviceName.unique()}
    for s in trace_q.keys():
        # anomal_span_df[anomal_span_df["serviceName"]=="frontendservice"].abnormal.sum() 

        score = int(anomal_span_df[anomal_span_df["serviceName"]==s].abnormal.sum())
        
        if s == "frontendservice":
            s = "frontend"
        
        service_dict[s] += ALPHA * score
    
    # print(json.dumps(service_dict, indent=2, sort_keys=True))
    rank_list = [(k, v) for k, v in service_dict.items()]
    rank_list.sort(key=lambda x: x[1], reverse=True)
    print(rank_list)



if __name__ == "__main__":
    main()