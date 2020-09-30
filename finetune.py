import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
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
from tqdm import tqdm

TRAIN_DATASET_PATH = '/data/nextcloud/dbc2017/files/jupyter/train_data'
IMG_SIZE = (512, 512)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_fns, label_dict, data_transforms):
        self.image_fns = image_fns
        self.label_dict= label_dict
        self.transforms = data_transforms
    
    def __getitem__(self, index):
        label = self.label_dict[image_fns[index].split('/')[-2]]
        image = Image.open(image_fns[index]).convert("RGB")
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

train_transform = transforms.Compose(
    [transforms.Scale(IMG_SIZE[0]),
     transforms.CenterCrop(IMG_SIZE[0]),
     transforms.RandomHorizontalFlip(p=0.5),
     #transforms.RandomAffine((-15, 15)),
     transforms.RandomRotation((-15, 15)),
     transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3),
     #transforms.RandomResizedCrop(IMG_SIZE),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

val_transform = transforms.Compose(
    [transforms.Scale(IMG_SIZE[0]),
     transforms.CenterCrop(IMG_SIZE[0]),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


train_fns, val_fns = train_test_split(image_fns, test_size=0.1, shuffle=True)

train_dataset = ImageDataset(train_fns, id_labels, train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                          shuffle=True, num_workers=16)
val_dataset = ImageDataset(val_fns, id_labels, val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                          shuffle=True, num_workers=8)
datalaoders_dict = {'train':train_loader, 'val':val_loader}

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
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
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()
model_net, scratch_hist = train_model(model, datalaoders_dict, criterion, optimizer, num_epochs=1)
torch.save(model_net.state_dict(),'/data/nextcloud/dbc2017/files/jupyter/model/res18_test.pth')

#torch.save(model.state_dict(), 'model_best.pth')
#model.load_state_dict(torch.load('model_best.pth'))
#model.to(device)