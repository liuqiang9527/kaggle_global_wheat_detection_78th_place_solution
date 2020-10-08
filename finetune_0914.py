import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import numpy as np
from tqdm import tqdm

TRAIN_DATASET_PATH = '../dataset/train_data'
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
class PadResize(object):
    def __init__(self,size):
        self.interpolation = Image.BILINEAR
        self.padding_v = [124, 116, 104]
        self.size = size
    def __call__(self, img):
        target_size = self.size
        padding_v = tuple(self.padding_v)
        interpolation = self.interpolation
        w, h = img.size
        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        return ret_img



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_fns, label_dict, data_transforms):
        self.image_fns = image_fns
        self.label_dict= label_dict
        self.transforms = data_transforms
    
    def __getitem__(self, index):
        label = self.label_dict[self.image_fns[index].split('/')[-2]]
        image = Image.open(self.image_fns[index]).convert("RGB")
        image = self.transforms(image)
        
        return image, label#, image_fns[index]
    
    def __len__(self):
        return len(self.image_fns)
    

        
image_fns = glob(os.path.join(TRAIN_DATASET_PATH, '*', '*.*'))
label_names = [s.split('/')[-2] for s in image_fns]
unique_labels = list(set(label_names))
unique_labels.sort()
id_labels = {_id:name for name, _id in enumerate(unique_labels)}

NUM_CLASSES = len(unique_labels)
print("NUM_CLASSES:", NUM_CLASSES)

train_transform = transforms.Compose([
     transforms.RandomRotation((-30, 30)),
#      transforms.RandomRotation(90),
#      transforms.RandomRotation(-90),
     PadResize(IMG_SIZE[0]),
#      transforms.Scale(IMG_SIZE[0]),
#      transforms.CenterCrop(IMG_SIZE[0]),
     transforms.ColorJitter(brightness=1, contrast=0.5,saturation=1),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

val_transform = transforms.Compose([
#     [transforms.Scale(IMG_SIZE[0]),
#      transforms.CenterCrop(IMG_SIZE[0]),
     PadResize(IMG_SIZE[0]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


train_fns, val_fns = train_test_split(image_fns, test_size=0.1, shuffle=True)

train_dataset = ImageDataset(train_fns, id_labels, train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           shuffle=True, 
                                           num_workers=20,
                                           drop_last=True)
val_dataset = ImageDataset(val_fns, id_labels, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                                         batch_size=BATCH_SIZE,
                                         shuffle=True, 
                                         num_workers=20,
                                         drop_last=True)
datalaoders_dict = {'train':train_loader, 'val':val_loader}

# from pyretri.models.backbone.backbone_impl.reid_baseline import ft_net_own
# path_file = '/data/nextcloud/dbc2017/files/jupyter/model/resnet50-19c8e357.pth'
# model = ft_net_own(progress=True)

from senet import Baseline
model = Baseline(3094,
                 1,
                 'se_resnext50_32x4d-a260b3a4.pth',
                 # 重要! ImageNet预训练权重下载地址为http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth
                 'bnneck',
                 'after',
                 'se_resnext50',
                 'imagenet'
                )


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
                       
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phaseSS
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)[0]
#                         print(outputs)
                        _, preds, = torch.max(outputs, 1)
#                         smooth_labels = smooth_one_hot(labels, NUM_CLASSES, smoothing=0.4)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # loss.backward()
                        # optimizer.step()
                       scaler.scale(loss).backward()
                       scaler.step(optimizer) 
                       scaler.update()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'models/se_resnext50_512_best.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = LabelSmoothingLoss(NUM_CLASSES, 0.4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model_net = train_model(model, datalaoders_dict, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
# torch.save(model_net.state_dict(),'models/se_resnext50_512_best.pth')
