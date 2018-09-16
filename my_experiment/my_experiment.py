from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt  
import torchvision
import numpy as np 
import torch.nn as nn

import os 
import random
import math

from my_generator import Omniglot_generator
from utils import show_image



LEARNING_RATE = 0.1


class DiffNet(torch.nn.Module):
    def __init__(self):
        super(DiffNet, self).__init__()
        
        self.fc1_linear_weights = nn.Parameter(torch.randn(128, 128))


        self.fc2_linear_weights = nn.Parameter(torch.randn(1,128))



        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

        # Attention!!!!!!!!!!!!!!!!!!!!!!


        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer5 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))


    def forward(self,x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
     
        input_size = out.size()[0]
        
        out = torch.squeeze(out)
        
        mat = torch.zeros(input_size, input_size)
        
        for i in range(input_size):
            for j in range(input_size):
                
                temp = torch.nn.functional.linear(torch.cat((out[i],out[j]),0), self.fc1_linear_weights)

                mat[i][j] = torch.nn.functional.linear(temp, self.fc2_linear_weights)

    
        
        out = torch.sigmoid(out)
        

        return mat

N_EPISODE = 100000

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    feature_encoder = DiffNet()

    feature_encoder.apply(weights_init)

    print('start training...')
    best_acc = 0.0
    train_dataset = Omniglot_generator('../datas/images_background',5,5)
    train_loader = DataLoader(train_dataset,batch_size = 1, shuffle = False)

    test_dataset = Omniglot_generator('../datas/images_evaluation',5,5,n_test = 15)
    test_loader = DataLoader(test_dataset,batch_size = 1, shuffle = False)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr = LEARNING_RATE)


    
    for episode in range(N_EPISODE):
        images,labels = train_loader.__iter__().next()
        images = images.permute(1,0,2,3)
        images = images.float()
        logit = feature_encoder(images)
        
        labels = torch.squeeze(labels)
        mat_len = labels.size()[0]
        objective = torch.zeros(mat_len, mat_len)
        for i in range(mat_len):
            for j in range(mat_len):
                if labels[i] == labels[j]:
                    objective[i][j] = 1
        
        

        mse = nn.MSELoss()
        loss = mse(objective, logit)


        #loss = torch.mean(torch.mean((objective - logit)**2,1 ),0)


        feature_encoder.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
 
        feature_encoder_optim.step()


        if (episode+1) % 100 == 0:
            print(loss)
            
            




        if (episode + 1) % 500 == 0:
            true_labels = 0
            total_labels = 10 * 15

            for step in range(10):
                images,labels = test_loader.__iter__().next()
                images = images.permute(1,0,2,3)
                images = images.float()
                logit = feature_encoder(images)
                labels = torch.squeeze(labels)
                mat_len = labels.size()[0]

                

                for i in range(25,mat_len):
            
                    if labels[i] == torch.argmax(logit[i][:25],dim = 0).int():
                        true_labels += 1
                
                 
                
            print(logit)        
            print('The accuracy is %f' %(true_labels / total_labels))





            



        
            
                



if __name__ == '__main__':
    main()




    








