import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import networkx
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
import numpy as np
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from RCAEval.io.time_series import drop_constant

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = enc_in
        
        #attention score
        self._attention = torch.ones(self.channels,1)
        self._attention = Variable(self._attention, requires_grad=False)
        self.fs_attention = torch.nn.Parameter(self._attention.data)

        self.IsTest = False
        self.pretrain = False
        self.project = False

        #encoder
        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        
        for i in range(self.channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
            self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        #decoder
        self.Decoder_Seasonal = nn.ModuleList()
        self.Decoder_Trend = nn.ModuleList()
        
        for i in range(self.channels):
            self.Decoder_Seasonal.append(nn.Linear(self.pred_len,self.seq_len))
            self.Decoder_Trend.append(nn.Linear(self.pred_len,self.seq_len))  

        self.Decoder_Seasonal_pointwise = nn.Linear(self.seq_len * self.channels, 1)
        self.Decoder_Trend_pointwise = nn.Linear(self.seq_len * self.channels, 1)

        #projector
        self.Proj_Seasonal = nn.ModuleList()
        self.Proj_Trend = nn.ModuleList()
        self.Proj_Seasonal_2 = nn.ModuleList()
        self.Proj_Trend_2 = nn.ModuleList()
        self.activation = nn.PReLU()
        for i in range(self.channels):
            self.Proj_Seasonal.append(nn.Linear(self.pred_len,self.pred_len * 2))
            self.Proj_Trend.append(nn.Linear(self.pred_len,self.pred_len * 2))
            self.Proj_Seasonal_2.append(nn.Linear(self.pred_len * 2,self.pred_len))
            self.Proj_Trend_2.append(nn.Linear(self.pred_len * 2,self.pred_len))
        

    def forward(self, x):
        if self.pretrain:
            x = x.transpose(1, 2)
   
            seasonal_init, trend_init = self.decompsition(x)

            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype)#.to("cuda:0")
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype)#.to("cuda:0")
            
            if self.project:

                for i in range(self.channels):
                    seasonal_output[:,i,:] = self.Proj_Seasonal_2[i](self.activation(self.Proj_Seasonal[i](self.Linear_Seasonal[i](seasonal_init[:,i,:].clone()))))
                    trend_output[:,i,:] = self.Proj_Trend_2[i](self.activation(self.Proj_Trend[i](self.Linear_Trend[i](trend_init[:,i,:].clone()))))

                x = seasonal_output + trend_output
            else:
                with torch.no_grad():
                    for i in range(self.channels):
                        seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:].clone())
                        trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:].clone())

                    x = seasonal_output + trend_output

            return x.transpose(1, 2)

        # x: [Batch, Input length, Channel]
        x = x.transpose(1, 2)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype)#to("cuda:0")
        trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype)#.to("cuda:0")
        
        seasonal_output_1 = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.seq_len],dtype=seasonal_init.dtype)#.to("cuda:0")
        trend_output_1 = torch.zeros([trend_init.size(0),trend_init.size(1),self.seq_len],dtype=trend_init.dtype)#.to("cuda:0")
        
        for i in range(self.channels):
            seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:].clone())
            trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:].clone())

        
        seasonal_output = seasonal_output *  F.softmax(self.fs_attention, dim=0)
        trend_output = trend_output * F.softmax(self.fs_attention, dim = 0)
        
        for i in range(self.channels):
            seasonal_output_1[:,i,:] = self.Decoder_Seasonal[i](seasonal_output[:,i,:].clone())
            trend_output_1[:,i,:] = self.Decoder_Trend[i](trend_output[:,i,:].clone())

        if self.IsTest:
            reshape_seasonal = torch.reshape(seasonal_output_1, (1, 1, 32*self.channels))
            reshape_trend = torch.reshape(trend_output_1, (1, 1, 32*self.channels))
        else:
            reshape_seasonal = torch.reshape(seasonal_output_1, (128, 1, 32*self.channels))
            reshape_trend = torch.reshape(trend_output_1, (128, 1, 32*self.channels))
        
        y1 = self.Decoder_Seasonal_pointwise(reshape_seasonal)
        y2 = self.Decoder_Trend_pointwise(reshape_trend)

        x = y1 + y2 
        x = x.transpose(1,2)

        return x

    def setPretrain(self, x):
        self.pretrain = x

    def setProj(self, x):
        self.project = x

    def setTest(self, x):
        self.IsTest = x
warnings.filterwarnings('ignore')

drop_num = 10

class Dataset_RCA(Dataset):
    def __init__(self, root_path, flag, 
                 data_path, target, 
                 features='MS', size=None, scale=True, timeenc=0, freq='s', train_only=False):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test']
        type_map = {'train': 0,'test': 1}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = drop_constant(df_raw)
        try:
            df_raw.drop("time", axis=1, inplace=True) #synthetic don't need this one
        except:
            pass

        data_len = len(df_raw)

        border1s = [0, int(data_len*(3/4)) - self.seq_len]
        border2s = [int(data_len*(3/4)), data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw

        train_data = df_data[border1:border2]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
def data_provider(args, flag):
    Data = Dataset_RCA
    timeenc = 0
    train_only = False

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = 's'

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = 128
        freq = 's'

    data_set = Data(
        flag=flag,
        size=[32, 0, 1],
        features='MS',
        target="",
        timeenc=timeenc,
        freq=freq,
        train_only=train_only,
        root_path=args.root_path,
        data_path=args.data_path,
        scale=True
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
def hierarchical_contrastive_loss(z1, z2, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if d >= temporal_unit:
            loss += temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    return loss / d

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    
    positive_pairs = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(positive_pairs, positive_pairs.transpose(1, 2))  # B x 2T x 2T
    
    positive_logits = sim[:, :T, T:]  
    positive_logits = -F.log_softmax(positive_logits, dim=-1)
    
    loss = positive_logits.mean()
    return loss

def pre_train(train_data, train_loader, model, optimizer, args, target_idx, cuda):
    train_steps = len(train_loader)
    model.train()
    total_batches = len(train_loader)
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        batch_x = batch_x.float()
        if cuda == "cuda:0":
            batch_x = batch_x.to("cuda:0")

        ts_l = batch_x.size(1)
        crop_l = 32
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=batch_x.size(0))

        model.setProj(True)
        out = model(take_per_row(batch_x, crop_offset + crop_eleft, 32))
        p1 = out[:, -crop_l:]
        out = model(take_per_row(batch_x, crop_offset + crop_left, 32))
        p2 = out[:, :crop_l]

        model.setProj(False)
        out = model(take_per_row(batch_x, crop_offset + crop_left, 32))
        z1 = out[:, :crop_l]
        out = model(take_per_row(batch_x, crop_offset + crop_eleft, 32))
        z2 = out[:, -crop_l:]

        loss = (hierarchical_contrastive_loss(p1,z2,temporal_unit=0) 
                + hierarchical_contrastive_loss(p2,z1,temporal_unit=0)) * 0.5
        loss.backward()
        optimizer.step()

    return loss

def train(train_data, train_loader, model, optimizer, args, target_idx, cuda):
    train_steps = len(train_loader)

    if cuda == "cuda:0":
        model.to("cuda:0")
    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):

        optimizer.zero_grad()
        if cuda == "cuda:0":
            batch_x = batch_x.float().to("cuda:0")
            batch_y = batch_y.float().to("cuda:0")
        else:
            batch_x = batch_x.float()
            batch_y = batch_y.float()

        outputs = model(batch_x)

        f_dim = -1 
        outputs = outputs[:, -1:, f_dim:]
        if cuda == "cuda:0":
            batch_y = batch_y[:, -1:, target_idx].to("cuda:0")
        else:
            batch_y = batch_y[:, -1:, target_idx]
        loss = F.mse_loss(outputs, batch_y)

        loss.backward()
        optimizer.step()

    attention = model.fs_attention
    return attention.data, loss

def test(test_data, test_loader, model, optimizer, args, target_idx, cuda):
    test_steps = len(test_loader)
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            optimizer.zero_grad()
            if cuda=="cuda:0":
                batch_x = batch_x.float().to("cuda:0")
                batch_y = batch_y.float().to("cuda:0")
            else:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
            outputs = model(batch_x)
            f_dim = -1 
            outputs = outputs[:, -1:, f_dim:]
            if cuda == "cuda:0":
                batch_y = batch_y[:, -1:, target_idx].to("cuda:0")
            else: 
                batch_y = batch_y[:, -1:, target_idx]
            loss = F.mse_loss(outputs, batch_y)
    return loss

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def GraphConstruct(target, cuda, epochs, lr, optimizername,  file, args):
    print(f"graph construct for {target} {file}")
    train_data, train_loader = data_provider(args, flag='train')
    test_data, test_loader = data_provider(args, flag='test')

    df_tmp = pd.read_csv(file)
    df_tmp = drop_constant(df_tmp)
    try:
        df_tmp.drop("time", axis=1, inplace=True) #synthetic don't need this one
    except:
        pass

    targetidx = df_tmp.columns.get_loc(target)  

    window_size = 32
    layers = 128
    model = DLinear(window_size, layers, len(df_tmp.columns))
    
    if cuda == "cuda:0":
        model.to("cuda:0")
    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)  

    model.setPretrain(True)
    pbar = trange(1, epochs+1, desc="pre train")
    for ep in pbar:
        pretrain_loss = pre_train(train_data, train_loader, model, optimizer, args, targetidx, cuda)
        pbar.set_postfix(pretrain_loss=pretrain_loss)

    model.setPretrain(False)   
    pbar = trange(1, epochs+1, desc="train and test")
    for ep in pbar:
        scores, train_loss = train(train_data, train_loader, model, optimizer, args, targetidx, cuda)
        model.setTest(True)
        test_loss = test(test_data, test_loader, model, optimizer, args, targetidx, cuda=cuda)
        model.setTest(False)
        pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
    
    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i] > 1:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i] < 1:
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)
        
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index<((len(s)-1)/2): 
                if index>0:
                    ind=index
                    break
        if ind < 0:
            ind = 0       
        potentials = indices[ : ind+1].tolist()
    edge_to_target = dict()
    for v in potentials:    
        edge_to_target[(targetidx, v)]=0
    
    return edge_to_target


def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("The lengths of the input variables must be the same.")
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x_sq = sum(x[i] ** 2 for i in range(n))
    sum_y_sq = sum(y[i] ** 2 for i in range(n))
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x_sq - sum_x ** 2) * (n * sum_y_sq - sum_y ** 2)) ** 0.5
    if denominator == 0:
        return 0
    correlation = numerator / denominator
    return correlation

def breaktie(pagerank, G, trigger_point):
    if trigger_point == "None":
        return pagerank
    
    rank = []
    tmp_rank = []
    last_score = 0    
    for cnt, (node, score) in enumerate(pagerank.items()):
        if last_score != score:
            if len(tmp_rank) == 0:
                last_score = score
                rank.append(node)
            else:
                ad = []
                for i in range(len(tmp_rank)):
                    try: 
                        distance = nx.shortest_path_length(G, source=trigger_point, target=node)
                    except nx.NetwrokXNoPath:
                        distance = 0
                    ad.append(distance)
                ad = np.array(ad)
                dis_rank = np.argsort(ad, reverse=True)
                for i in range(len(dis_rank)):
                    rank.append(tmp_rank[dis_rank[i]])
                tmp_rank = [node]
        else:
            tmp_rank.append(node)
            if cnt == len(pagerank)-1:
                ad = []
                for i in range(len(tmp_rank)):
                    try: 
                        distance = nx.shortest_path_length(G, source=trigger_point, target=node)
                    except nx.NetwrokXNoPath:
                        distance = 0
                    ad.append(distance)
                ad = np.array(ad)
                dis_rank = np.argsort(ad, reverse=True)
                for i in range(len(dis_rank)):
                    rank.append(tmp_rank[dis_rank[i]])
    return rank


def Run(datafile, args):
    df_data = pd.read_csv(datafile)
    df_data = drop_constant(df_data)
    edges = dict()
    
    try:
        df_data.drop("time", axis=1, inplace=True) #synthetic don't need this one
    except:
        pass
    columns = list(df_data)

    for c in columns: 
        idx = df_data.columns.get_loc(c)
        edge = GraphConstruct(c, cuda=args.cuda, epochs=args.epochs, 
        lr=args.learning_rate, optimizername=args.optimizer, file=datafile, args=args)

        print(c, idx, edge)
        edges.update(edge)
    return edges, columns

def CreateGraph(edge, columns):
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in edge:
        p1,p2 = pair
        G.add_edge(columns[p2], columns[p1])
    return G


def main(datafiles):
    edge_pair, columns = Run(datafiles) 
    pruning = pd.read_csv(args.root_path + '/' + args.data_path)   
    pruning = drop_constant(pruning)
    try:
        pruning.drop("time", axis=1, inplace=True) #synthetic don't need this one
    except:
        pass
    G = CreateGraph(edge_pair, columns)

    while not nx.is_directed_acyclic_graph(G):
        edge_cor = []
        edges = G.edges()
        for edge in edges:
            source, target = edge
            edge_cor.append(pearson_correlation(pruning[source], pruning[target]))
        tmp = np.array(edge_cor)
        tmp_idx = np.argsort(tmp)
        edges = list(edges)
        source, target= edges[tmp_idx[0]][0], edges[tmp_idx[0]][1]

        G.remove_edge(source, target)
 
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    personalization = {}
    for node in G.nodes():
        if node in dangling_nodes:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.5
    pagerank = nx.pagerank(G, personalization=personalization)
    pagerank = dict(sorted(pagerank.items(), key=lambda x: x[1], reverse=True))
    

def run(data, inject_time=None, dataset=None, with_bg=False, args=None, **kwargs):
    # set args
    args.cuda = -1
    args.epochs = 1
    args.learning_rate = 0.001
    args.optimizer = 'Adam'
    args.num_workers = 8
    args.root_cause = "unknown"

    nrepochs = args.epochs
    learningrate = args.learning_rate
    optimizername = args.optimizer
    cuda=args.cuda
    root_cause = args.root_cause

    data_path = os.path.join(args.root_path, args.data_path)
    edge_pair, columns = Run(data_path, args) 
    pruning = pd.read_csv(data_path)   
    pruning = drop_constant(pruning)
    try: 
        pruning.drop("time", axis=1, inplace=True) #synthetic don't need this one
    except:
        pass

    G = CreateGraph(edge_pair, columns)

    while not nx.is_directed_acyclic_graph(G):
        edge_cor = []
        edges = G.edges()
        for edge in edges:
            source, target = edge
            edge_cor.append(pearson_correlation(pruning[source], pruning[target]))
        tmp = np.array(edge_cor)
        tmp_idx = np.argsort(tmp)
        edges = list(edges)
        source, target= edges[tmp_idx[0]][0], edges[tmp_idx[0]][1]

        G.remove_edge(source, target)
 
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    personalization = {}
    for node in G.nodes():
        if node in dangling_nodes:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.5
    pagerank = nx.pagerank(G, personalization=personalization)
    ranks = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    ranks = [r[0] for r in ranks]
    return {
        "ranks": ranks
    }
    

if __name__ == "__main__":

    args = argparse.Namespace()
    args.root_path = os.getcwd()
    args.data_path = "data/sock-shop-1/carts_cpu/1/data.csv"

    data = pd.read_csv(os.path.join(args.root_path, args.data_path))
    data = drop_constant(data)
    try:
        data.drop("time", axis=1, inplace=True) #synthetic don't need this one
    except:
        pass

    output = run(data, inject_time=None, dataset="sock-shop", args=args)
    print(output)