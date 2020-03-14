# First let's import some necessities
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

DANCE_NUM = 0
EAT_NUM = 0
HUG_NUM = 0
KICK_NUM = 0
RUN_NUM = 0
SWIM_NUM = 0
Train_num = 10
NUM_LIST = [DANCE_NUM, EAT_NUM, HUG_NUM, KICK_NUM, RUN_NUM, SWIM_NUM]
'''
This part is left for data loading and transporting
The input should be the raw feature extracted by the dense trajectories and the out put should be the N*D [feature] and [N*1 label]

Output:
features: N*D
labels: N*1
'''
makeran = lambda s: np.random.permutation(s)
train_index = []
for i in range(len(NUM_LIST)):
    if i != 0: train_index.append(makeran(NUM_LIST[i])[Train_num]+NUM_LIST[i-1])
    if i == 0: train_index.append(makeran(NUM_LIST[i])[Train_num])
test_index = set(np.linspace(0,len(features)-1,len(features)))^set(train_index)
train_features = features[train_index]
train_labels = labels[train_index]
test_features = features[test_index]
test_labels = labels[test_index]
'''
Output right now is train_feature, train_label, test_feature and test_label
'''
#Now I am writing the feature and label to the db.

def write_db(db_type, db_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos() #Create varient features and labels
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])
        ])
        transaction.put(
            'train_%02d'.format(i),
            feature_and_label.SerializaToString()
        )
        #Close the transaction and then close the db
        del transaction
        del db
write_db("minidb", "pattern_train.minidb", train_features, train_labels) #the type is minidb, the db name is pattern_train.minidb
write_db("minidb", "pattern_test.minidb", test_features, test_labels)

#Now I am creating the neural network named pattern_test1, I am loading the data into the net using TensorProtosDBInput function
net_proto = core.Net("pattern_train")
dbreader = net_proto.CreateDB([], "dbreader", db="pattern_train.minidb", db_type="minidb")
data_unit, label = net_proto.TensorProtosDBInput([dbreader], ["data_unit","label"], batch_size=16)
data = model.Cast(data_unit8, "data", to=core.DataType.FLOAT)
data = model.Scale(data, data, scale=float(1./256))
#The train input and output is data, and label respectively

# Function to construct a MLP neural network
# The input 'model' is a model helper and 'data' is the input data blob's name
def AddMLPModel(model, data):
    size = data.shape[1]
    sizes = [size, size * 2, size * 2, 10]
    layer = data
    for i in range(len(sizes) - 1):
        layer = brew.fc(model, layer, 'dense_{}'.format(i), dim_in=sizes[i], dim_out=sizes[i + 1])
        layer = brew.relu(model, layer, 'relu_{}'.format(i))
    softmax = brew.softmax(model, layer, 'softmax')
    return softmax

#I can print the net to see if things are going well, the input is dbreader, the output should be X, Y
print("The net looks like this:")
print(str(net_proto.Proto()))

workspace.CreateNet(net_proto)#I am putting my network into the workspace
