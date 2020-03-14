import torch
import torch.nn as nn
from mypath import Path

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()
        # # 设置关键帧权值的参数
        # # 输入[batch, 3, 10, 112, 112]
        # self.conv_1 = nn.Conv3d(3, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # # 输出[batch, 3, 10, 56, 56]

        # self.conv_2 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool_2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # # 输出[batch, 16, 10, 28, 28]
        # # 首先对数据维度交换 输出[batch, 10, 16, 28, 28] permute(0, 2, 1, 3, 4)
        # # 随后将输出改为[batch, 10, 16*28*28]
        # self.fc_1 = nn.Linear(16*28*28, 4*28*28)
        # self.fc_2 = nn.Linear(4*28*28, 1)

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #output: [batchsize, 128, 5, 28, 28]
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        # x1 = x.clone()
        # x1 = self.relu(self.conv_1(x1))
        # x1 = self.pool_1(x1)
        # x1 = self.relu(self.conv_2(x1))
        # x1 = self.pool_2(x1)
        # x1 = self.dropout(x1)
        # x1 = x1.permute(0, 2, 1, 3, 4)
        # x1 = x1.reshape([-1, 10, 12544])

        # x1 = self.relu(self.fc_1(x1))
        # x1 = self.fc_2(x1)#输出是[batch, 10, 1]
        # x1 = x1.view(-1, 10)
        # k = x1.clone()
        # x1 = nn.Softmax(dim=1)(k)#Softmax 一下
        # #这里应该要算一个sparsity loss
        # temp = x.clone()
        # batch = len(temp)
        # for i in range(batch):
        #   for j in range(10):
        #     x[i,:,j] = temp[i,:,j]*x1[i,j]
        
            
        # print(x.size())
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        # return logits, sum(sum(x1))
        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    # b = [model.conv_1, model.pool_1, model.conv_2, model.pool_2, model.fc_1, model.fc_2, 
    # model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
    #      model.conv5a, model.conv5b, model.fc6, model.fc7]
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k
def l1_penalty(var):
  totalsum = 0
  for i in var:
    totalsum += torch.abs(i).sum()
  return totalsum

def l2_penalty(var):
  return torch.sqrt(torch.pow(var,2).sum())

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())