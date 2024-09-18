import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform, train):
        self.transform = transform
        self.train = train
        if self.train:
            self.image_paths = glob.glob('/home/yuudaitori/test/data/cifar-10/train/**/*.png', recursive=True)
        else:
            self.image_paths = glob.glob('/home/yuudaitori/test/data/cifar-10/test/**/*.png', recursive=True)
        self.data_num = len(self.image_paths)

    def __len__(self):
        return self.data_num
    
    def __getitem__(self, idx):
        p = self.image_paths[idx]
        label = int(os.path.basename(os.path.dirname(p)))
        image = Image.open(p)

        if self.transform:
            image = self.transform(image)

        return image,label
    

# 以下確認用------------------------------------------    
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_set = MyDataset(transform, train=True)
print(data_set[0])