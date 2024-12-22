import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse
import pandas as pd
import os, sys
import math
import scipy
#import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import spatial
import itertools as it
import string
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(
                self.input_channels[i], self.hidden_channels[i],
                self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize, hidden=self.hidden_channels[i],
                        shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)



def attention(ConvLstm_out):
    attention_w = []
    for k in range(5):
        attention_w.append(torch.sum(torch.mul(ConvLstm_out[k], ConvLstm_out[-1]))/5)
    m = nn.Softmax()
    attention_w = torch.reshape(m(torch.stack(attention_w)), (-1, 5))
    cl_out_shape = ConvLstm_out.shape
    ConvLstm_out = torch.reshape(ConvLstm_out, (5, -1))
    convLstmOut = torch.matmul(attention_w, ConvLstm_out)
    convLstmOut = torch.reshape(convLstmOut, (cl_out_shape[1], cl_out_shape[2], cl_out_shape[3]))
    return convLstmOut

class CnnEncoder(nn.Module):
    def __init__(self, in_channels_encoder):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, (2, 2), 1),
            nn.SELU()
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, (2, 2), 0),
            nn.SELU()
        )   
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, (2, 2), 0),
            nn.SELU()
        )
    def forward(self, X):
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out


class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(
            input_channels=32, hidden_channels=[32], kernel_size=3,
            step=5, effective_step=[4])
        self.conv2_lstm = ConvLSTM(
            input_channels=64, hidden_channels=[64], kernel_size=3,
            step=5, effective_step=[4])
        self.conv3_lstm = ConvLSTM(
            input_channels=128, hidden_channels=[128],
            kernel_size=3, step=5, effective_step=[4])
        self.conv4_lstm = ConvLSTM(
            input_channels=256, hidden_channels=[256],
            kernel_size=3, step=5, effective_step=[4])

    def forward(self, conv1_out, conv2_out, 
                conv3_out, conv4_out):
        conv1_lstm_out = self.conv1_lstm(conv1_out)
        conv1_lstm_out = attention(conv1_lstm_out[0][0])
        conv2_lstm_out = self.conv2_lstm(conv2_out)
        conv2_lstm_out = attention(conv2_lstm_out[0][0])
        conv3_lstm_out = self.conv3_lstm(conv3_out)
        conv3_lstm_out = attention(conv3_lstm_out[0][0])
        conv4_lstm_out = self.conv4_lstm(conv4_out)
        conv4_lstm_out = attention(conv4_lstm_out[0][0])
        return conv1_lstm_out.unsqueeze(0), conv2_lstm_out.unsqueeze(0), conv3_lstm_out.unsqueeze(0), conv4_lstm_out.unsqueeze(0)

class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0),
            nn.SELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 2, 1, 1),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 1, 1, 0),
            nn.SELU()
        )
    
    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        deconv4 = self.deconv4(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim = 1)
        deconv3 = self.deconv3(deconv4_concat)
        # print(f"{deconv3.shape=}")
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim = 1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim = 1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1


class MSCRED(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder):
        super(MSCRED, self).__init__()
        self.cnn_encoder = CnnEncoder(in_channels_encoder)
        self.conv_lstm = Conv_LSTM()
        self.cnn_decoder = CnnDecoder(in_channels_decoder)
    
    def forward(self, x):
        # x.shape = 5,3,64,64
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(x)
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out = \
            self.conv_lstm(conv1_out, conv2_out, conv3_out, conv4_out)

        gen_x = self.cnn_decoder(conv1_lstm_out, conv2_lstm_out, 
                                conv3_lstm_out, conv4_lstm_out)
        return gen_x



# parameters
ts_type = "node" # type of time series data, "node" or "link"
step_max = 5     # maximum step in ConvLSTM
gap_time = 10    # tride width 
win_size = [10, 30, 60]  # window size of each segment
min_time = 0     # minimum time point
max_time = 20000 # maximum time point

train_start = 0  # train start point
train_end = 8000  # train end point
test_start = 8000  # test start point
test_end = 20000  # test end point
# raw_data_path = '/opt/home/s3967801/Pytorch-MSCRED/data/synthetic_data_with_anomaly-s-1.csv'  # path to load raw data
# save_data_path = '/opt/home/s3967801/Pytorch-MSCRED/data/'  # path to save data

# create temporary directory for save_data_path
from tempfile import TemporaryDirectory
save_data_path = TemporaryDirectory().name

ts_colname="agg_time_interval"
agg_freq='5min'

matrix_data_path = save_data_path + "matrix_data/"
if not os.path.exists(matrix_data_path):
    os.makedirs(matrix_data_path)


def generate_signature_matrix_node():
	data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)
	sensor_n = data.shape[0]
	# min-max normalization
	max_value = np.max(data, axis=1)
	min_value = np.min(data, axis=1)
	data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)
	data = np.transpose(data)

	#multi-scale signature matix generation
	for w in range(len(win_size)):
		matrix_all = []
		win = win_size[w]
		print ("generating signature with window " + str(win) + "...")
		for t in range(min_time, max_time, gap_time):
			#print t
			matrix_t = np.zeros((sensor_n, sensor_n))
			if t >= 60:
				for i in range(sensor_n):
					for j in range(i, sensor_n):
						#if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
						matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
						matrix_t[j][i] = matrix_t[i][j]
			matrix_all.append(matrix_t)
		path_temp = matrix_data_path + "matrix_win_" + str(win)
		np.save(path_temp, matrix_all)
		del matrix_all[:]

	print ("matrix generation finish!")

def generate_train_test_data():
	#data sample generation
	print ("generating train/test data samples...")
	matrix_data_path = save_data_path + "matrix_data/"

	train_data_path = matrix_data_path + "train_data/"
	if not os.path.exists(train_data_path):
		os.makedirs(train_data_path)
	test_data_path = matrix_data_path + "test_data/"
	if not os.path.exists(test_data_path):
		os.makedirs(test_data_path)

	data_all = []
	for w in range(len(win_size)):
		path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + ".npy"
		data_all.append(np.load(path_temp))

	train_test_time = [[train_start, train_end], [test_start, test_end]]
	for i in range(len(train_test_time)):
		for data_id in range(int(train_test_time[i][0]/gap_time), int(train_test_time[i][1]/gap_time)):
			#print data_id
			step_multi_matrix = []
			for step_id in range(step_max, 0, -1):
				multi_matrix = []
				# for k in range(len(value_colnames)):
				for i in range(len(win_size)):
					multi_matrix.append(data_all[i][data_id - step_id])
				step_multi_matrix.append(multi_matrix)

			if data_id >= (train_start/gap_time + win_size[-1]/gap_time + step_max) and data_id < (train_end/gap_time): # remove start points with invalid value
				path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)
			elif data_id >= (test_start/gap_time) and data_id < (test_end/gap_time):
				path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)

			#print np.shape(step_multi_matrix)

			del step_multi_matrix[:]

	print ("train/test data generation finish!")



def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum,n = 0.0, 0
        for x in tqdm(dataLoader):
            x = x.to(device)
            x = x.squeeze()
            #print(type(x))
            l = torch.mean((model(x)-x[-1].unsqueeze(0))**2)
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            # print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
            
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, train_l_sum/n))

def test(dataLoader, model):
    print("------Testing-------")
    index = 800
    loss_list = []
    reconstructed_data_path = "./data/matrix_data/reconstructed_data/"
    with torch.no_grad():
        for x in dataLoader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x) 
            path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            # l = criterion(reconstructed_matrix, x[-1].unsqueeze(0)).mean()
            # loss_list.append(l)
            # print("[test_index %d] [loss: %f]" % (index, l.item()))
            index += 1



def mscred(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSCRED(3, 256)

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    # select only n time series
    # n = 64
    #data = data.iloc[:, :n+1]
    # round up the number of columns to the nearest 2^n, add addition columns with zeros
    n = len(normal_df.columns)
    n = 2 ** math.ceil(math.log2(n))
    if n > len(normal_df.columns):
        for i in range(n - len(normal_df.columns)):
            normal_df[f"empty_{i}"] = 0
            anomal_df[f"empty_{i}"] = 0

    col_name_map = {i: col for i, col in enumerate(normal_df.columns)}


    normal_data = np.array(normal_df, dtype=np.float64)
    anomal_data = np.array(anomal_df, dtype=np.float64)
    sensor_n = normal_data.shape[1]
    # min-max normalization 
    max_value = np.max(normal_data, axis=0)
    min_value = np.min(normal_data, axis=0)
    normal_data = (normal_data - min_value)/(max_value - min_value + 1e-6) 
    anomal_data = (anomal_data - min_value)/(max_value - min_value + 1e-6)
    
    # multi-scale signature matix generation for normal data
    matrix_all = {}
    for win in win_size:
        # print ("generating signature with window " + str(win) + "...")
        normal_matrix = []
        for t in range(0, normal_data.shape[0], 10):
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(normal_data[i, t - win:t], normal_data[j, t - win:t])/(win) # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            normal_matrix.append(matrix_t)
        matrix_all[win] = normal_matrix

    # multi-scale signature matix generation for abnormal data
    abnormal_matrix_all = {}
    for win in win_size:
        #print ("generating signature with window " + str(win) + "...")
        abnormal_matrix = []
        for t in range(0, anomal_data.shape[0], 10):
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(anomal_data[i, t - win:t], anomal_data[j, t - win:t])/(win)

            abnormal_matrix.append(matrix_t) 
        abnormal_matrix_all[win] = abnormal_matrix

    # prepare train data from normal data
    step_multi_matrix_all = []
    for data_id in range(0, 30):
        step_multi_matrix = []
        for step_id in range(step_max, 0, -1):
            multi_matrix = []
            for win in win_size:
                multi_matrix.append(matrix_all[win][data_id - step_id])
            step_multi_matrix.append(multi_matrix)

        step_multi_matrix_all.append(step_multi_matrix)
    
    # prepare test data from abnormal data
    step_multi_matrix_all_abnormal = []
    num_epoch = 10
    for data_id in range(0, num_epoch):
        step_multi_matrix = []
        for step_id in range(step_max, 0, -1):
            multi_matrix = []
            for win in win_size:
                multi_matrix.append(abnormal_matrix_all[win][data_id - step_id])
            step_multi_matrix.append(multi_matrix)

        step_multi_matrix_all_abnormal.append(step_multi_matrix)


    # PREPARE training DATALOADER
    train_data = torch.tensor(step_multi_matrix_all, dtype=torch.float32)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    # PREPARE test DATALOADER
    test_data = torch.tensor(step_multi_matrix_all_abnormal, dtype=torch.float32)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # TRAINING
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
    # print("------training on {}-------".format(device))
    num_epoch = 10
    for epoch in range(num_epoch):
        train_l_sum, n = 0.0, 0
        # for x in tqdm(train_loader):
        for x in train_loader:
            x = x.to(device)
            x = x.squeeze()
            l = torch.mean((model(x)-x[-1].unsqueeze(0))**2)
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            # print(f"[Epoch {epoch}/10][Batch {n}/{len(train_loader)}] [loss: {l.item()}]")

        # print(f"[Epoch {epoch+1}/{num_epoch}] [loss: {train_l_sum/n:.5f}]")

    # RECONSTRUCT NORMAL DATA
    with torch.no_grad():
        constructed_test_matrixes = []
        for test_matrix in test_loader:
            test_matrix = test_matrix.to(device)
            test_matrix = test_matrix.squeeze()
            reconstructed_matrix = model(test_matrix)
            constructed_test_matrixes.append(reconstructed_matrix.cpu().detach().numpy().squeeze()[0])
    
    # evaluate reconstruction error
    error_list = []
    for gt_matrix, constructed_matrix in zip(step_multi_matrix_all_abnormal, constructed_test_matrixes):
        gt_matrix = gt_matrix[-1][0]

        matrix_error = np.square(np.subtract(gt_matrix, constructed_matrix))
        threshold = 0.005
        num_broken = len(matrix_error[matrix_error > threshold])

        if num_broken > 0:
            # print("Anomaly detected")
            error_list = [(i, error) for i, error in enumerate(np.sum(matrix_error, axis=1))]

            error_list = sorted(error_list, key=lambda x: x[1], reverse=True)
            
            # replace index by column name
            error_list = [(col_name_map[i], error) for i, error in error_list]
            break
    if not error_list:
        # generate random error list
        error_list = [(col_name_map[i], np.random.rand()) for i in range(sensor_n)]

    error_list = sorted(error_list, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in error_list if "empty" not in x[0]]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSCRED(3, 256)

    # Read online boutique data
    metric_path = "/opt/home/s3967801/cfm/data/mm-ob/currencyservice_cpu/1/simple_metrics.csv"
    log_path = "/opt/home/s3967801/cfm/data/mm-ob/currencyservice_cpu/1/logts.csv"
    trace_lat_path = "/opt/home/s3967801/cfm/data/mm-ob/currencyservice_cpu/1/tracets_lat.csv"
    trace_err_path = "/opt/home/s3967801/cfm/data/mm-ob/currencyservice_cpu/1/tracets_err.csv"
    inject_time_path = "/opt/home/s3967801/cfm/data/mm-ob/currencyservice_cpu/1/inject_time.txt"

    data = pd.read_csv(metric_path)
    with open(inject_time_path) as f:
        inject_time = int(f.readlines()[0].strip())
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    # select only n time series
    n = 64
    data = data.iloc[:, :n+1]
   
    
    length = 10
    normal_df = data[data["time"] < inject_time].tail(10 * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(10 * 60 // 2)
    # anomal_df = pd.concat([normal_df, anomal_df], axis=0)


    # drop time 
    normal_df = normal_df.drop(columns=["time"])
    anomal_df = anomal_df.drop(columns=["time"])

    col_name_map = {i: col for i, col in enumerate(normal_df.columns)}

    normal_data = np.array(normal_df, dtype=np.float64)
    anomal_data = np.array(anomal_df, dtype=np.float64)
    sensor_n = normal_data.shape[1]
    # min-max normalization 
    max_value = np.max(normal_data, axis=0)
    min_value = np.min(normal_data, axis=0)
    normal_data = (normal_data - min_value)/(max_value - min_value + 1e-6) 
    anomal_data = (anomal_data - min_value)/(max_value - min_value + 1e-6)
    
    # multi-scale signature matix generation for normal data
    matrix_all = {}
    for win in win_size:
        print ("generating signature with window " + str(win) + "...")
        normal_matrix = []
        for t in range(0, normal_data.shape[0], 10):
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(normal_data[i, t - win:t], normal_data[j, t - win:t])/(win) # rescale by win
                        matrix_t[j][i] = matrix_t[i][j]
            normal_matrix.append(matrix_t)
        matrix_all[win] = normal_matrix

    # multi-scale signature matix generation for abnormal data
    abnormal_matrix_all = {}
    for win in win_size:
        print ("generating signature with window " + str(win) + "...")
        abnormal_matrix = []
        for t in range(0, anomal_data.shape[0], 10):
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(anomal_data[i, t - win:t], anomal_data[j, t - win:t])/(win)

            abnormal_matrix.append(matrix_t) 
        abnormal_matrix_all[win] = abnormal_matrix

    # prepare train data from normal data
    step_multi_matrix_all = []
    for data_id in range(0, 30):
        step_multi_matrix = []
        for step_id in range(step_max, 0, -1):
            multi_matrix = []
            for win in win_size:
                multi_matrix.append(matrix_all[win][data_id - step_id])
            step_multi_matrix.append(multi_matrix)

        step_multi_matrix_all.append(step_multi_matrix)
    
    # prepare test data from abnormal data
    step_multi_matrix_all_abnormal = []
    num_epoch = 10
    for data_id in range(0, num_epoch):
        step_multi_matrix = []
        for step_id in range(step_max, 0, -1):
            multi_matrix = []
            for win in win_size:
                multi_matrix.append(abnormal_matrix_all[win][data_id - step_id])
            step_multi_matrix.append(multi_matrix)

        step_multi_matrix_all_abnormal.append(step_multi_matrix)


    # PREPARE training DATALOADER
    train_data = torch.tensor(step_multi_matrix_all, dtype=torch.float32)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    # PREPARE test DATALOADER
    test_data = torch.tensor(step_multi_matrix_all_abnormal, dtype=torch.float32)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # TRAINING
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
    print("------training on {}-------".format(device))
    num_epoch = 10
    for epoch in range(num_epoch):
        train_l_sum, n = 0.0, 0
        for x in tqdm(train_loader):
            x = x.to(device)
            x = x.squeeze()
            l = torch.mean((model(x)-x[-1].unsqueeze(0))**2)
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            # print(f"[Epoch {epoch}/10][Batch {n}/{len(train_loader)}] [loss: {l.item()}]")

        print(f"[Epoch {epoch+1}/{num_epoch}] [loss: {train_l_sum/n:.5f}]")

    # RECONSTRUCT NORMAL DATA
    with torch.no_grad():
        constructed_test_matrixes = []
        for test_matrix in test_loader:
            test_matrix = test_matrix.to(device)
            test_matrix = test_matrix.squeeze()
            reconstructed_matrix = model(test_matrix)
            constructed_test_matrixes.append(reconstructed_matrix.cpu().detach().numpy().squeeze()[0])
    
    # evaluate reconstruction error
    for gt_matrix, constructed_matrix in zip(step_multi_matrix_all_abnormal, constructed_test_matrixes):
        gt_matrix = gt_matrix[-1][0]

        print(f"{gt_matrix.shape=}")
        print(f"{constructed_matrix.shape=}")

        matrix_error = np.square(np.subtract(gt_matrix, constructed_matrix))
        threshold = 0.005
        num_broken = len(matrix_error[matrix_error > threshold])
        print(f"{num_broken=}")

        if num_broken > 0:
            print("Anomaly detected")
            error_list = [(i, error) for i, error in enumerate(np.sum(matrix_error, axis=1))]

            error_list = sorted(error_list, key=lambda x: x[1], reverse=True)
            
            # replace index by column name
            error_list = [(col_name_map[i], round(error)) for i, error in error_list]
            print(error_list[:5])
            exit(0)



