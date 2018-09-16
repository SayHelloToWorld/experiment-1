#encode:utf-8
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt  
import torchvision

import os 
import random
import numpy as np 

from utils import split_integer


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def get_default_transform():
    return transforms.Compose([Rotate(random.choice([0,90,180,270]))])


class Omniglot_generator(Dataset):
    def __init__(self,data_folder, n_classes, n_samples, n_test = 0, transform = get_default_transform()):
        self.n_test = n_test
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.folder_names = []
        for alpha_folder in os.listdir(data_folder):
            for character_folder in os.listdir(data_folder + '/' + alpha_folder):
                
                self.folder_names.append(data_folder+'/'+alpha_folder + '/' + character_folder + '/')

        self.transform = transform



    def __getitem__(self, idx):

        images = []
        if self.n_test == 0:
            test_split = []
        else:
            test_split = split_integer(self.n_test,self.n_classes)

        #train images
        for class_names in random.sample(self.folder_names,self.n_classes):
            filenames_list = os.listdir(class_names)
            for filename in random.sample(filenames_list, self.n_samples):
                img = Image.open(class_names + filename)
                img = self.transform(img)
                img = img.resize((28,28), resample=Image.LANCZOS)
                images.append(np.array(img, dtype = np.double))

        #test images
        if self.n_test != 0:
            for i,class_names in enumerate(random.sample(self.folder_names,self.n_classes)):
                filenames_list = os.listdir(class_names)
                for filename in random.sample(filenames_list, test_split[i]):
                    img = Image.open(class_names + filename)
                    img = self.transform(img)
                    img = img.resize((28,28), resample=Image.LANCZOS)
                    images.append(np.array(img, dtype = np.double))

        
        #train labels
        labels = []
        for i in range(self.n_classes):
            for j in range(self.n_samples):
                labels.append(i)
        #test labels
        if self.n_test != 0:
            for i in range(self.n_classes):
                for j in range(test_split[i]):
                    labels.append(i) 

        data = []
        for i in range(self.n_classes*self.n_samples + self.n_test):
            data.append((images[i],labels[i]))
        random.shuffle(data) 
        for i in range(self.n_classes*self.n_samples + self.n_test):
            images[i],labels[i] = data[i]

        images = np.asarray(images)
        labels = np.asarray(labels)

        return images, labels


    def __len__(self):
        return 100000

from utils import show_image


def unit_test():
    
    dataset = Omniglot_generator('../datas/images_background',5,5)
    dataloader = DataLoader(dataset,batch_size = 1, shuffle = False)

    for images, labels in dataloader:
        print(images.size())
        print(labels.size())
        show_image(images[0][3])
        print(labels)

if __name__ == '__main__':
    unit_test()

        



