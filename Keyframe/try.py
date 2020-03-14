import numpy as np
import os, sys
from PIL import Image
#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import init


train_dir='train_1'
test_dir='test'
#test1_dir='video1-1'
#test2_dir='video1-2'
#test3_dir='video1-3'
#test4_dir='video1-4'
#test5_dir='video1-5'





train_transforms=transforms.Compose([transforms.Resize((64,64)),transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),  (0.229,0.224,0.225))])
test_transforms=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),  (0.229,0.224,0.225))])
train_data=datasets.ImageFolder(train_dir,train_transforms)
test_data=datasets.ImageFolder(test_dir,test_transforms)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=32)


#test1_data=datasets.ImageFolder(test1_dir,test_transforms)
#test2_data=datasets.ImageFolder(test2_dir,test_transforms)
#test3_data=datasets.ImageFolder(test3_dir,test_transforms)
#test4_data=datasets.ImageFolder(test4_dir,test_transforms)
#test5_data=datasets.ImageFolder(test5_dir,test_transforms)

#test1_loader=torch.utils.data.DataLoader(test1_data,batch_size=32)
#test2_loader=torch.utils.data.DataLoader(test2_data,batch_size=32)
#test3_loader=torch.utils.data.DataLoader(test3_data,batch_size=32)
#test4_loader=torch.utils.data.DataLoader(test4_data,batch_size=32)
#test5_loader=torch.utils.data.DataLoader(test5_data,batch_size=32)







class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(3, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            #nn.Dropout(p=0.5),
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            #nn.Dropout(p=0.5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 14 * 14, 120),
            #nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 20),
            #nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(20, 6)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多


netR = ResNet()

#netR.train()

learning_rate=0.00035
criterion = nn.CrossEntropyLoss()
#criterion.cuda("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(netR.parameters(), lr=learning_rate, betas = (0.9, 0.999), eps=1e-08, weight_decay = 0.17)


def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs  # 设置学习次数
    print_every = print_every
    steps = 0
    netR.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 优化器梯度清零

            # 前馈及反馈
            outputs = model(inputs)  # 数据前馈，正向传播
            loss = criterion(outputs, labels)  # 输出误差
            loss.backward()  # 误差反馈
            optimizer.step()  # 优化器更新参数

            running_loss += loss.item()

            if steps % print_every == 0:
                # test the accuracy

                print('EPOCHS : {}/{}'.format(e + 1, epochs),
                      'Loss : {:.4f}'.format(running_loss / print_every))
                #accuracy_test(model, validloader)


deep_learning(netR,train_loader,100,40,criterion,optimizer,torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def accuracy_test(model, dataloader):
    correct = 0
    total = 0
    #num1=0
    #num2=0
    #num3=0
    #num4=0
    #num5=0
    #num6=0
    #model.cuda()  # 将模型放入GPU计算，能极大加快运算速度
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            #images, labels = images.to('cuda'), labels.to('cuda')
            #print(images.shape)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            #num1 += (predicted == 0).sum().item()
            #num2 += (predicted == 1).sum().item()
            #num3 += (predicted == 2).sum().item()
           # num4 += (predicted == 3).sum().item()
           # num5 += (predicted == 4).sum().item()
           # num6 += (predicted == 5).sum().item()
            # 将预测及标签两相同大小张量逐一比较各相同元素的个数
    print('the accuracy is {:.4f}'.format(correct / total))
 #   print(correct , total)
    #print('num1=%d'%(num1))
    #print('num2=%d'%(num2))
    #print('num3=%d'%(num3))
    #print('num4=%d'%(num4))
    #print('num5=%d'%(num5))
    #print('num6=%d'%(num6))


accuracy_test(netR, train_loader)
accuracy_test(netR,test_loader)

#accuracy_test(netR,test1_loader)
#accuracy_test(netR,test2_loader)
#accuracy_test(netR,test3_loader)
#accuracy_test(netR,test4_loader)
#accuracy_test(netR,test5_loader)

torch.save(netR, 'model.pth')