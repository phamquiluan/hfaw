# refactor from https://github.com/IntelligentDDS/MicroRank
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

from tqdm import tqdm
import numpy as np
from dateutil.parser import parse
import pandas as pd


def pageRank(p_ss, p_sr, p_rs, v, operation_length, trace_length, d=0.85, alpha=0.01):
    iteration = 25
    service_ranking_vector = np.ones(
        (operation_length, 1)) / float(operation_length + trace_length)
    request_ranking_vector = np.ones(
        (trace_length, 1)) / float(operation_length + trace_length)

    for i in range(iteration):
        updated_service_ranking_vector = d * \
            (np.dot(p_sr, request_ranking_vector) +
             alpha * np.dot(p_ss, service_ranking_vector))
        updated_request_ranking_vector = d * \
            np.dot(p_rs, service_ranking_vector) + (1.0 - d) * v
        service_ranking_vector = updated_service_ranking_vector / \
            np.amax(updated_service_ranking_vector)
        request_ranking_vector = updated_request_ranking_vector / \
            np.amax(updated_request_ranking_vector)

    normalized_service_ranking_vector = service_ranking_vector / \
        np.amax(service_ranking_vector)
    return normalized_service_ranking_vector



def trace_pagerank(operation_operation, operation_trace, trace_operation, pr_trace, anomaly):
    """ Calculate pagerank weight of anormaly_list or normal_list
    :arg 
    :return
        operation weight:
        weight[operation][0]: operation
        weight[operation][1]: weight
    """
    operation_length = len(operation_operation)
    trace_length = len(operation_trace)

    p_ss = np.zeros((operation_length, operation_length), dtype=np.float32)
    p_sr = np.zeros((operation_length, trace_length), dtype=np.float32)
    p_rs = np.zeros((trace_length, operation_length), dtype=np.float32)

    # matrix = np.zeros((n, n), dtype=np.float32)
    pr = np.zeros((trace_length, 1), dtype=np.float32)

    node_list = []
    for key in operation_operation.keys():
        node_list.append(key)

    trace_list = []
    for key in operation_trace.keys():
        trace_list.append(key)

    # matrix node*node
    for operation in operation_operation:
        child_num = len(operation_operation[operation])

        for child in operation_operation[operation]:
            p_ss[node_list.index(child)][node_list.index(
                operation)] = 1.0 / child_num

    # matrix node*request
    for trace_id in operation_trace:
        child_num = len(operation_trace[trace_id])
        for child in operation_trace[trace_id]:
            p_sr[node_list.index(child)][trace_list.index(trace_id)] \
                = 1.0 / child_num

    # matrix request*node
    for operation in trace_operation:
        child_num = len(trace_operation[operation])

        for child in trace_operation[operation]:
            p_rs[trace_list.index(child)][node_list.index(operation)] \
                = 1.0 / child_num

    kind_list = np.zeros(len(trace_list))
    p_srt = p_sr.T
    for i in range(len(trace_list)):
        index_list = [i]
        if kind_list[i] != 0:
            continue
        n = 0
        for j in range(i, len(trace_list)):
            if (p_srt[i] == p_srt[j]).all():
                index_list.append(j)
                n += 1
        for index in index_list:
            kind_list[index] = n

    num_sum_trace = 0
    kind_sum_trace = 0
    if not anomaly:
        for trace_id in pr_trace:
            num_sum_trace += 1.0 / kind_list[trace_list.index(trace_id)]
        for trace_id in pr_trace:
            pr[trace_list.index(trace_id)] = 1.0 / \
                kind_list[trace_list.index(trace_id)] / num_sum_trace
    else:
        for trace_id in pr_trace:
            kind_sum_trace += 1.0 / kind_list[trace_list.index(trace_id)]
            num_sum_trace += 1.0 / len(pr_trace[trace_id])
        for trace_id in pr_trace:
            pr[trace_list.index(trace_id)] = 1.0 / (kind_list[trace_list.index(trace_id)] / kind_sum_trace * 0.5
                                                    + 1.0 / len(pr_trace[trace_id])) / num_sum_trace * 0.5

    if anomaly:
        print("\nAnomaly_PageRank:")
    else:
        print("\nNormal_PageRank:")
    result = pageRank(p_ss, p_sr, p_rs, pr, operation_length, trace_length)

    weight = {}
    sum = 0
    for operation in operation_operation:
        sum += result[node_list.index(operation)][0]

    trace_num_list = {}
    for operation in operation_operation:
        trace_num_list[operation] = 0
        i = node_list.index(operation)
        for j in range(len(trace_list)):
            if p_sr[i][j] != 0:
                trace_num_list[operation] += 1

    for operation in operation_operation:
        weight[operation] = result[node_list.index(
            operation)][0] * sum / len(operation_operation)

    # for score in sorted(weight.items(), key=lambda x: x[1], reverse=True):
    #     print("%-50s: %.5f" % (score[0], score[1]))

    return weight, trace_num_list



def anomaly_detection(start_time, end_time, slo, operation_list):
    """ Input short time trace data and calculate the expect_duration.
    expect_duration = operation1 * mean_duration1 + variation_duration1 +
                     operation2 * mean_duration2 + variation_duration2
    if expect_duration < real_duration  error                 
    :arg
        date: format 2020-08-14 or 2020-08-*
        start_time end_time  expect 30s or 1min traces
    :return
        if error_rate > 1%:
           return True    
    """

    span_list = get_span(start_time, end_time)
    if len(span_list) == 0:
        print("Error: Current span list is empty ")
        return False
    operation_count = get_operation_duration_data(operation_list, span_list)

    anormaly_trace = 0
    total_trace = 0
    for trace_id in operation_count:
        total_trace += 1
        real_duration = float(operation_count[trace_id]["duration"]) / 1000.0
        expect_duration = 0.0
        for operation in operation_count[trace_id]:
            if "duration" == operation:
                continue
            expect_duration += operation_count[trace_id][operation] * (
                slo[operation][0] + 1.5 * slo[operation][1])

        if real_duration > expect_duration:
            anormaly_trace += 1

    print("anormaly_trace", anormaly_trace)
    print("total_trace", total_trace)
    print()
    if anormaly_trace > 8:
        anormaly_rate = float(anormaly_trace) / total_trace
        print("anormaly_rate", anormaly_rate)
        return True

    else:
        return False


def trace_anormaly_detect(operation_list, slo):
    """ Determine single trace state
    :arg
        operation_list: operation_count[traceid] # list of operation of single trace
        slo: slo list
    
    :return
         if real_duration > expect_duration:
             return True
         else:
             return False    
    """
    expect_duration = 0.0
    real_duration = float(operation_list["duration"]) / 1000.0
    for operation in operation_list:
        if operation == "duration":
            continue
        expect_duration += operation_list[operation] * \
            (slo[operation][0] + slo[operation][1])

    if real_duration > expect_duration + 50:
        return True
    else:
        return False


def trace_list_partition(operation_count, slo):
    """
    Partition all the trace list in operation_count to normal_list and abnormal_list
    :arg
        operation_count: all the trace operation
        operation_count[traceid][operation] = 1
    :return
        normal_list: normal traceid list
        abnormal_list: abnormal traceid list
       
    """
    normal_list = []  # normal traceid list
    abnormal_list = []  # abnormal traceid list
    for traceid in operation_count:
        normal = trace_anormaly_detect(
            operation_list=operation_count[traceid], slo=slo)
        if normal:
            abnormal_list.append(traceid)
        else:
            normal_list.append(traceid)

    return abnormal_list, normal_list



def get_operation_duration_data(operation_list, span_list):
    """ Query the operation and duration in span_list for anormaly detector 
    :arg
        operation_list: contains all operation
        operation_dict:  { "operation1": 1, "operation2":2 ... "operationn": 0, "duration": 666}
        span_list: all the span_list in one anomaly detection interval (1 min or 30s)
    :return
        { 
           traceid: {
               operation1: 1
               operation2: 2
           }
        }
    """

    operation_dict = {}

    trace_id = span_list[0]["_source"]["traceID"]

    def server_client_determined():
        for tag in doc["tags"]:
            if tag["key"] == "span.kind":
                return tag["value"]

    def get_operation_name():
        operation_name_tmp = doc["operationName"]
        operation_name_tmp = operation_name_tmp.split("/")[-1]
        operation_name_tmp = doc["process"]["serviceName"] + \
            "_" + operation_name_tmp
        return operation_name_tmp

    def init_dict(trace_id):
        if trace_id not in operation_dict:
            operation_dict[trace_id] = {}
            for operation in operation_list:
                operation_dict[trace_id][operation] = 0
            operation_dict[trace_id]["duration"] = 0

    length = 0
    for doc in span_list:
        doc = doc["_source"]
        tag = server_client_determined()
        operation_name = get_operation_name()

        init_dict(doc["traceID"])

        if trace_id == doc["traceID"]:
            operation_dict[trace_id][operation_name] += 1
            length += 1

            if doc["process"]["serviceName"] == "frontend" and tag == "server":
                operation_dict[trace_id]["duration"] += doc["duration"]

        else:
            if operation_dict[trace_id]["duration"] == 0:
                if length > 45:
                    operation_dict.pop(trace_id)

                else:
                    operation_dict.pop(trace_id)

            trace_id = doc["traceID"]
            length = 0
            operation_dict[trace_id][operation_name] += 1

            if doc["process"]["serviceName"] == "frontend" and tag == "server":
                operation_dict[trace_id]["duration"] += doc["duration"]

    return operation_dict




def timestamp(datetime):
    timeArray = time.strptime(str(datetime), "%Y-%m-%d %H:%M:%S")
    ts = int(time.mktime(timeArray)) * 1000
    # print(ts)
    return ts



def calculate_spectrum_without_delay_list(
    anomaly_result,
    normal_result,
    anomaly_list_len,
    normal_list_len,
    top_max,
    normal_num_list,
    anomaly_num_list,
    spectrum_method
):
    spectrum = {}

    for node in anomaly_result:
        spectrum[node] = {}
        # spectrum[node]["ef"] = anomaly_result[node] * anomaly_list_len
        # spectrum[node]["nf"] = anomaly_list_len - anomaly_result[node] * anomaly_list_len
        spectrum[node]["ef"] = anomaly_result[node] * anomaly_num_list[node]
        spectrum[node]["nf"] = anomaly_result[node] * \
            (anomaly_list_len - anomaly_num_list[node])
        if node in normal_result:
            #spectrum[node]["ep"] = normal_result[node] * normal_list_len
            #spectrum[node]["np"] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]["ep"] = normal_result[node] * normal_num_list[node]
            spectrum[node]["np"] = normal_result[node] * \
                (normal_list_len - normal_num_list[node])
        else:
            spectrum[node]["ep"] = 0.0000001
            spectrum[node]["np"] = 0.0000001

    for node in normal_result:
        if node not in spectrum:
            spectrum[node] = {}
            #spectrum[node]["ep"] = normal_result[node] * normal_list_len
            #spectrum[node]["np"] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]["ep"] = (
                1 + normal_result[node]) * normal_num_list[node]
            spectrum[node]["np"] = normal_list_len - normal_num_list[node]
            if node not in anomaly_result:
                spectrum[node]["ef"] = 0.0000001
                spectrum[node]["nf"] = 0.0000001

    # print("\n Micro Rank Spectrum raw:")
    # print(json.dumps(spectrum))
    result = {}

    for node in spectrum:
        # Dstar2
        if spectrum_method == "dstar2":
            result[node] = spectrum[node]["ef"] * spectrum[node]["ef"] / \
                (spectrum[node]["ep"] + spectrum[node]["nf"])
        # Ochiai
        elif spectrum_method == "ochiai":
            result[node] = spectrum[node]["ef"] / \
                math.sqrt((spectrum[node]["ep"] + spectrum[node]["ef"]) * (
                    spectrum[node]["ef"] + spectrum[node]["nf"]))

        elif spectrum_method == "jaccard":
            result[node] = spectrum[node]["ef"] / (spectrum[node]["ef"] + spectrum[node]["ep"]
                                                   + spectrum[node]["nf"])

        elif spectrum_method == "sorensendice":
            result[node] = 2 * spectrum[node]["ef"] / \
                (2 * spectrum[node]["ef"] + spectrum[node]
                 ["ep"] + spectrum[node]["nf"])

        elif spectrum_method == "m1":
            result[node] = (spectrum[node]["ef"] + spectrum[node]
                            ["np"]) / (spectrum[node]["ep"] + spectrum[node]["nf"])

        elif spectrum_method == "m2":
            result[node] = spectrum[node]["ef"] / (2 * spectrum[node]["ep"] + 2 * spectrum[node]["nf"] +
                                                   spectrum[node]["ef"] + spectrum[node]["np"])
        elif spectrum_method == "goodman":
            result[node] = (2 * spectrum[node]["ef"] - spectrum[node]["nf"] - spectrum[node]["ep"]) / \
                (2 * spectrum[node]["ef"] + spectrum[node]
                 ["nf"] + spectrum[node]["ep"])
        # Tarantula
        elif spectrum_method == "tarantula":
            result[node] = spectrum[node]["ef"] / (spectrum[node]["ef"] + spectrum[node]["nf"]) / \
                (spectrum[node]["ef"] / (spectrum[node]["ef"] + spectrum[node]["nf"]) +
                 spectrum[node]["ep"] / (spectrum[node]["ep"] + spectrum[node]["np"]))
        # RussellRao
        elif spectrum_method == "russellrao":
            result[node] = spectrum[node]["ef"] / \
                (spectrum[node]["ef"] + spectrum[node]["nf"] +
                 spectrum[node]["ep"] + spectrum[node]["np"])

        # Hamann
        elif spectrum_method == "hamann":
            result[node] = (spectrum[node]["ef"] + spectrum[node]["np"] - spectrum[node]["ep"] - spectrum[node]
                            ["nf"]) / (spectrum[node]["ef"] + spectrum[node]["nf"] + spectrum[node]["ep"] + spectrum[node]["np"])

        # Dice
        elif spectrum_method == "dice":
            result[node] = 2 * spectrum[node]["ef"] / \
                (spectrum[node]["ef"] + spectrum[node]
                 ["nf"] + spectrum[node]["ep"])

        # SimpleMatching
        elif spectrum_method == "simplematcing":
            result[node] = (spectrum[node]["ef"] + spectrum[node]["np"]) / (spectrum[node]
                                                                            ["ef"] + spectrum[node]["np"] + spectrum[node]["nf"] + spectrum[node]["ep"])

        # RogersTanimoto
        elif spectrum_method == "rogers":
            result[node] = (spectrum[node]["ef"] + spectrum[node]["np"]) / (spectrum[node]["ef"] +
                                                                            spectrum[node]["np"] + 2 * spectrum[node]["nf"] + 2 * spectrum[node]["ep"])

    # Top-n节点列表
    top_list = []
    score_list = []
    for index, score in enumerate(sorted(result.items(), key=lambda x: x[1], reverse=True)):
        if index < top_max + 6:
            top_list.append(score[0])
            score_list.append(score[1])
            #print("%-50s: %.8f" % (score[0], score[1]))
    return top_list, score_list


def online_anomaly_detect_RCA(slo, operation_list):
    while True:
        current_time = datetime.datetime.strptime(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")-datetime.timedelta(minutes=1)

        start_time = current_time - datetime.timedelta(seconds=60)
        anormaly_flag = anomaly_detection(start_time=timestamp(start_time),
                                               end_time=timestamp(current_time), slo=slo, operation_list=operation_list)
        if anormaly_flag:
            detect_time = current_time
            start_time = detect_time - datetime.timedelta(seconds=5)
            end_time = detect_time + datetime.timedelta(seconds=55)
            year = str(time.strptime(str(detect_time),
                       "%Y-%m-%d %H:%M:%S").tm_year)
            mon = time.strptime(str(detect_time), "%Y-%m-%d %H:%M:%S").tm_mon
            day = time.strptime(str(detect_time), "%Y-%m-%d %H:%M:%S").tm_mday
            hour = time.strptime(str(detect_time), "%Y-%m-%d %H:%M:%S").tm_hour
            minute = time.strptime(
                str(detect_time), "%Y-%m-%d %H:%M:%S").tm_min
            if mon > 9:
                mon = str(mon)
            else:
                mon = "0" + str(mon)

            if day > 9:
                day = str(day)
            else:
                day = "0" + str(day)

            if minute >= 1:
                if hour > 9:
                    hour = str(hour)
                else:
                    hour = "0" + str(hour)
            else:
                if hour - 1 > 9:
                    hour = hour - 1
                    hour = str(hour)
                elif hour == 0:
                    hour = "23"
                    current_day = time.strptime(
                        str(detect_time), "%Y-%m-%d %H:%M:%S").tm_mday
                    current_day = current_day - 1
                    if current_day > 9:
                        day = str(current_day)
                    else:
                        day = "0" + str(current_day)
                else:
                    hour = hour - 1
                    hour = "0" + str(hour)
            # date = year + "-" + mon + "-" + day
            # print("checkpoint", date)

            middle_span_list = get_span(
                timestamp(start_time), timestamp(end_time))
            operation_count = get_operation_duration_data(
                operation_list, middle_span_list)
            anomaly_list, normal_list = trace_list_partition(
                operation_count=operation_count, slo=slo)

            print("anomaly_list", len(anomaly_list))
            print("normal_list", len(normal_list))
            print("total", len(normal_list) + len(anomaly_list))

            if len(anomaly_list) == 0 or len(normal_list) == 0:
                continue
            operation_operation, operation_trace, trace_operation, pr_trace \
                = get_pagerank_graph(normal_list, middle_span_list)

            normal_trace_result, normal_num_list = trace_pagerank(
                operation_operation,
                operation_trace,
                trace_operation, 
                pr_trace,
                False
            )

            a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace \
                = get_pagerank_graph(anomaly_list, middle_span_list)
            anomaly_trace_result, anomaly_num_list = trace_pagerank(a_operation_operation, a_operation_trace,
                                                                    a_trace_operation, a_pr_trace,
                                                                    True)
            top_list, score_list = calculate_spectrum_without_delay_list(anomaly_result=anomaly_trace_result,
                                                                         normal_result=normal_trace_result,
                                                                         anomaly_list_len=len(
                                                                             anomaly_list),
                                                                         normal_list_len=len(
                                                                             normal_list),
                                                                         top_max=5,
                                                                         anomaly_num_list=anomaly_num_list,
                                                                         normal_num_list=normal_num_list,
                                                                         spectrum_method="dstar2")
            print(top_list, score_list)
            # sleep 5min after a fault
            time.sleep(240)
        time.sleep(60)


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



def get_pagerank_graph(df):
    """
    Query the pagerank graph
    
    :return
        operation_operation 存储子节点 Call graph
        operation_operation[operation_name] = [operation_name1 , operation_name1 ] 

        operation_trace 存储trace经过了哪些operation, 右上角 coverage graph
        operation_trace[traceid] = [operation_name1 , operation_name2]

        trace_operation 存储 operation被哪些trace 访问过, 左下角 coverage graph
        trace_operation[operation_name] = [traceid1, traceid2]  
        
        pr_trace: 存储trace id 经过了哪些operation，不去重
        pr_trace[traceid] = [operation_name1 , operation_name2]
    """
    operation_operation = {}
    operation_trace = {}
    trace_operation = {}
    pr_trace = {}
    
    # loop df
    parent_operation = {}
    op_dict = {}  # spanid -> ops
    par_dict = {}  # spanid -> parent_spanid
    child_dict = {}  # spanid -> child_spanids
    for idx, row in tqdm(df.iterrows()):
        traceid = row["traceID"]
        spanid = row["spanID"]
        op = row["operation"]
        parent_span_id = row["parentSpanID"]

        op_dict[spanid] = op
        par_dict[spanid] = parent_span_id
        if parent_span_id not in child_dict:
            child_dict[parent_span_id] = []
        child_dict[parent_span_id].append(spanid)

        # operation_trace[traceid] = []
        if traceid not in operation_trace:
            operation_trace[traceid] = []
        operation_trace[traceid].append(op)

        if op not in trace_operation:
            trace_operation[op] = []
        trace_operation[op].append(traceid)

    for idx, row in tqdm(df.iterrows()):
        spanid = row["spanID"]
        op = row["operation"]
        parent_span_id = row["parentSpanID"]

        if op not in operation_trace:
            operation_operation[op] = []

        
        # append child operation 
        if spanid in child_dict and len(child_dict[spanid]) > 0:
            operation_operation[op].extend([op_dict[child_spanid] for child_spanid in child_dict[spanid]])

    pr_trace = deepcopy(operation_trace)
    
    for k, v in operation_operation.items():
        operation_operation[k] = list(set(v))
    for k, v in operation_trace.items():
        operation_trace[k] = list(set(v))
    for k, v in trace_operation.items():
        trace_operation[k] = list(set(v))
    
    return operation_operation, operation_trace, trace_operation, pr_trace


def microrank(data, inject_time=None, dataset=None, **kwargs):
    # span_df = pd.read_csv("./data/mm-ob/checkoutservice_delay/1/traces.csv")
    span_df = data
    span_df["methodName"] = span_df["methodName"].fillna(span_df["operationName"])
    span_df["operation"] = span_df["serviceName"] + "_" + span_df["methodName"]

    # inject_time = int(inject_time) * 1_000_000  # convert from seconds to microseconds

    normal_df  = span_df[span_df["startTime"] + span_df["duration"] < inject_time]
    normal_slo = get_operation_slo(normal_df)
    normal_traceid = normal_df["traceID"].unique()

    anomal_df  = span_df[span_df["startTime"] + span_df["duration"] >= inject_time]
    
    normal_df["mean"] = normal_df["operation"].apply(lambda op: normal_slo[op]["mean"])
    normal_df["std"] = normal_df["operation"].apply(lambda op: normal_slo[op]["std"])

    anomal_df["mean"] = anomal_df["operation"].apply(lambda op: normal_slo[op]["mean"])
    anomal_df["std"] = anomal_df["operation"].apply(lambda op: normal_slo[op]["std"])

    normal_traces_df = anomal_df[anomal_df["duration"] / 1_000 < anomal_df["mean"] + 3 * anomal_df["std"]]
    anomal_traces_df = anomal_df[anomal_df["duration"] / 1_000 >= anomal_df["mean"] + 3 * anomal_df["std"]]

    normal_traceid = normal_traces_df["traceID"].unique()
    anomal_traceid = anomal_traces_df["traceID"].unique()
    
    operation_operation, operation_trace, trace_operation, pr_trace \
        = get_pagerank_graph(normal_df)

    normal_trace_result, normal_num_list = trace_pagerank(
        operation_operation,
        operation_trace,
        trace_operation,
        pr_trace,
        False
    )

    a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace \
        = get_pagerank_graph(anomal_df)

    anomaly_trace_result, anomaly_num_list = trace_pagerank(
        a_operation_operation,
        a_operation_trace,
        a_trace_operation,
        a_pr_trace,
        True
    )

    top_list, score_list = calculate_spectrum_without_delay_list(
        anomaly_result=anomaly_trace_result,
        normal_result=normal_trace_result,
        anomaly_list_len=len(anomal_traceid),
        normal_list_len=len(normal_traceid),
        top_max=5,
        anomaly_num_list=anomaly_num_list,
        normal_num_list=normal_num_list,
        spectrum_method="dstar2"
    )
    # print(top_list, score_list)
    return {
        "ranks": top_list,
    }
  


if __name__ == "__main__":
    main()