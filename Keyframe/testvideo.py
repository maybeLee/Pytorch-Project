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



model=torch.load('model_new5.pth')
#model.eval()


def process_image(image):
    ''' 对图片进行缩放，建材，标准化，并输出一个NUMPY数组
    '''

    # 调整图片大小
    pic = Image.open(image)
    pic=pic.resize((64,64))
    np_image=transforms.ToTensor()(pic)
    np_image=transforms.Normalize((0.485,0.456,0.406),  (0.229,0.224,0.225))(np_image)

    # 从图片中心抠出224 *224 的图像

    #pic = pic.crop([pic.size[0] / 2 - 112, pic.size[1] / 2 - 112, pic.size[0] / 2 + 112, pic.size[1] / 2 + 112])

    # 将图片转化为numpy数组
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = np.array(pic)
    np_image = np_image / 255

    for i in range(2):  # 使用和训练集同样的参数对图片进行数值标准化

        np_image[:, :, i] -= mean[i]
        np_image[:, :, i] /= std[i]

    np_image = np_image.transpose((2, 0, 1))  # PyTorch 要求颜色通道为第一个维度，但是在 PIL 图像和 Numpy 数组中是第三个维度，所以调整
    np_image = torch.from_numpy(np_image)  # 转化为张量
    np_image = np_image.float()
    #print(np_image.type)
    '''
    return np_image


def predict(image_path, model):
    ''' 预测图片.
    '''
    img = process_image(image_path)
    img = img.unsqueeze(0)   # 将图片多增加一维
    #print(img.shape)
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    return predicted






#file_dir='new'
train_dir='train_end'
test_dir='new'





"""
for dirpath, dirnames, filenames in os.walk(file_dir,topdown=False):

    for dirname in dirnames:

        real_class=dirname

        for dirpath1, dirnames1, filenames1 in os.walk(os.path.join(dirpath, dirname),topdown=False):

            right_num=0
            wrong_num=0
            totol_number=0

            for dirname1 in dirnames1:

                for dirpath2, dirnames2, filenames2 in os.walk(os.path.join(dirpath1, dirname1),topdown=False):

                    addnum=np.zeros(6)
                    for filename2 in filenames2:

                        now_img = os.path.join(dirpath2, filename2)
                        result=predict(now_img,model)
                        addnum[result]+=1

                    addlist=addnum.tolist()
                    max_index = addlist.index(max(addlist))
                    if (max_index==int(real_class)-1) :
                        right_num+=1
                        totol_number+=1
                        print(dirpath2,'right')
                    else :
                        wrong_num+=1
                        totol_number+=1
                        print(dirpath2,'wrong')
            print("class : %s  right_number : %d  wrong_number : %d  totol_number :%d" %(real_class,right_num,wrong_num,totol_number))
"""



def result(file_dir):
    right_all = 0
    wrong_all = 0
    all_all = 0
    right_video=0
    wrong_video=0
    all_video=0
    for f in os.listdir(file_dir):

        real_class = f
        now_file = os.path.join(file_dir, f)
        right_num = 0
        wrong_num = 0
        totol_number = 0
        for f1 in os.listdir(now_file):

            now_file1 = os.path.join(now_file, f1)
            addnum = np.zeros(6)

            for f2 in os.listdir(now_file1):

                now_img = os.path.join(now_file1, f2)
                result = predict(now_img, model)
                if result + 1 == int(real_class):
                    right_all += 1
                    all_all += 1
                else:
                    wrong_all += 1
                    all_all += 1
                addnum[result] += 1

            addlist = addnum.tolist()
            max_index = addlist.index(max(addlist))
            if (addnum[max_index] == addnum[int(real_class) - 1]):
                right_num += 1
                totol_number += 1
                # print(now_file1, 'right')
            else:
                wrong_num += 1
                totol_number += 1
                #print(now_file1, 'wrong')
                #print("wrong to class :%s" % (max_index + 1))
                #print(addnum)
        print("class : %s  right_number : %d  wrong_number : %d  totol_number :%d  right_tate : %4f" % (
        real_class, right_num, wrong_num, totol_number,float(right_num/totol_number)))
        right_video+=right_num
        wrong_video+=wrong_num
        all_video+=totol_number

    print(right_video, wrong_video, all_video, float(right_video / all_video))


result(train_dir)
result(test_dir)



