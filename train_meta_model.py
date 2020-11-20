from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from torch.autograd import Variable as V
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1:]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        # x = x.view(x.size(0), -1)
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out

class Net(nn.Module):
    def __init__(self , resnet, net):
        super(Net, self).__init__()
        self.resnet_layer = resnet
        self.net = net

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)
        return x

def train_model(model, criterion, optimizer, num_epochs=100):
    task_num = 4
    optim_step = 5
    spearman_test = 0
    criterion.cuda()
    model.cuda()
    meta_model = copy.deepcopy(model)
    temp_model = copy.deepcopy(model)

    for epoch in range(num_epochs):
        optimizer = exp_lr_scheduler(optimizer, epoch)

        print('############# train phase ###############')
        count = 0
        list_ind = list(range(165))
        np.random.shuffle(list_ind)
        for index in list_ind:

            if count % task_num == 0:
                name_to_param = dict(temp_model.named_parameters())
                for name, param in meta_model.named_parameters():
                    diff = param.data - name_to_param[name].data
                    name_to_param[name].data.add_(diff)
            name_to_param = dict(model.named_parameters())
            for name, param in temp_model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            ##meta-training
            dataloader_train, dataloader_valid = load_data('train', index)
            dataiter = iter(enumerate(dataloader_valid))
            model.train()  # Set model to training mode

            #first-level
            for batch_idx, data in enumerate(dataloader_train):
                inputs = data['image']
                batch_size = inputs.size()[0]
                labels = data['rating'].view(batch_size, -1)
                labels = labels / 5.0
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                for iii in range(optim_step):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                # second-level
                idx, data_val = next(dataiter)
                if idx >= len(dataloader_valid) - 1:
                    dataiter = iter(enumerate(dataloader_valid))
                inputs_val = data_val['image']
                batch_size1 = inputs_val.size()[0]
                labels_val = data_val['rating'].view(batch_size1, -1)
                labels_val = labels_val / 5.0
                if use_gpu:
                    try:
                        inputs_val, labels_val = Variable(inputs_val.float().cuda()), Variable(labels_val.float().cuda())
                    except:
                        print(inputs_val, labels_val)
                else:
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

                for iii in range(optim_step):
                    optimizer.zero_grad()
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)
                    loss_val.backward()
                    optimizer.step()

            name_to_param = dict(meta_model.named_parameters())
            for name, param in model.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff/task_num)
            count += 1

    torch.save(model.cuda(),
            'Personality_Model/FlickrAES_Personal_Meta_Resnet18.pt')


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    decay_rate =  0.9**(epoch // lr_decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train', worker_idx = 0):

    meta_num = 80
    data_dir = os.path.join('Image_dataset/Flickr_AES_Project/data')
    if mod == 'train':
        workers = pd.read_csv(os.path.join(data_dir,  'train_workers.csv'), sep=' ')
    else:
        workers = pd.read_csv(os.path.join(data_dir,  'test_workers.csv'), sep=' ')
    worker_orignal = pd.read_csv(os.path.join(data_dir,  'image_labeled_by_each_worker.csv'), sep=',')

    workers_fold = "Workers_Meta/"
    if not os.path.exists(workers_fold):
        os.makedirs(workers_fold)

    worker = workers['worker'].unique()[worker_idx]
    print("----worker number: %2d---- %s" %(worker_idx, worker))
    num_images = worker_orignal[worker_orignal['worker'].isin([worker])].shape[0]

    if mod == 'train':
        if num_images >= 110:
            bsize = meta_num
        else:
            bsize = int(num_images * 0.8)
    else:
        bsize = meta_num
    percent = bsize / num_images

    images = worker_orignal[worker_orignal['worker'].isin([worker])][[' imagePair', ' score']]

    train_dataframe, valid_dataframe = train_test_split(images, train_size=percent)
    train_path = workers_fold + "train_scores_" + worker + ".csv"
    test_path = workers_fold + "test_scores_" + worker + ".csv"
    train_dataframe.to_csv(train_path, sep=',', index=False)
    valid_dataframe.to_csv(test_path, sep=',', index=False)

    output_size = (224, 224)
    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='Image_dataset/Flickr_AES_Project/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='Image_dataset/Flickr_AES_Project/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))

    bsize_test = int(bsize / 4.0)
    dataloader_train = DataLoader(transformed_dataset_train, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
    dataloader_valid = DataLoader(transformed_dataset_valid, batch_size= bsize_test,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
    return dataloader_train, dataloader_valid


epochs = 50
net1 = models.resnet18(pretrained=True)
# modules = list(net1.children())[:-1]
# net1 = torch.nn.Sequential(*modules)
# net2 = BaselineModel1(1, 0.5, 512)
fc_features = net1.fc.out_features
net2 = BaselineModel1(1, 0.5, fc_features)
model_ft = Net(resnet=net1, net=net2)

ignored_params = list(map(id, model_ft.net.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model_ft.parameters())
optimizer = optim.Adam([
    {'params': base_params},
    {'params': model_ft.net.parameters(), 'lr': 1e-3}
], lr=1e-4)

criterion = nn.MSELoss()

train_model(model_ft, criterion, optimizer, num_epochs=epochs)
