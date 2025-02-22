from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

label_name = ["airplane", "automobile", "bird",
              "cat", "deer", "dog",
              "frog", "horse", "ship", "truck"]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx

def default_loader(path):
    return Image.open(path).convert("RGB")

# train_transform = transforms.Compose([
#     transforms.RandomCrop(28),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(90),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    # transforms.RandomGrayscale(0.2),
    # transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# train_transform = transforms.Compose([
#     transforms.RandomCrop(28),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     # transforms.RandomRotation(90),
#     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
#     transforms.RandomGrayscale(0.2),
#     transforms.ToTensor()
# ])
#
# test_transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor()
# ])

class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None,
                 loader = default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for im_item in im_list:
            #"/home/kuan/dataset/CIFAR10/TRAIN/" \
            #"airplane/aeroplane_s_000021.png"
            im_label_name = im_item.split("\\")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)

im_train_list = glob.glob("D:\DeepLearning\pythonProject2-Cifar10\cifar-10-batches-py\TRAIN/*/*.png")
im_test_list = glob.glob("D:\DeepLearning\pythonProject2-Cifar10\cifar-10-batches-py\TEST/*/*.png")

train_dataset = MyDataset(im_train_list,
                         transform=train_transform)
test_dataset = MyDataset(im_test_list,
                        transform =test_transform)

train_loader = DataLoader(dataset=train_dataset,
                               batch_size=128,
                               shuffle=True,
                               num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                               batch_size=128,
                               shuffle=False,
                               num_workers=4)   # num_workers的值和模型训练快慢有关，和训练出的模型的performance无关

#print("num_of_train", len(train_dataset))
#print("num_of_test", len(test_dataset))









