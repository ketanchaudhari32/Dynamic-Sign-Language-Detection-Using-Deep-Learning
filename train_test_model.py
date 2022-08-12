#importing libraries

import numpy as np
import glob
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from spatio_temporal_conv import R2Plus1DClassifier
import torch
from torch import nn, optim

num_classes = 10

class_labels = np.load('./class_{}_dataset/class_labels.npy'.format(num_classes))

label_map = {label:index for index,label in enumerate(class_labels)}
label_map

#loading training,testing and validation data
X_train = []
Y_train = []
X_val = []
Y_val = []
X_test = []
Y_test = []

for file in glob.glob('./class_{}_features/*/*.npy'.format(num_classes)):    
    file_path = file.split('/')
    split = file_path[6]
    file_name = file_path[7].split('.')[0].split('_')[0]

    data = np.load(file)

    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    data = data.transpose((3, 0, 1, 2))
    

    if split == 'train':
        X_train.append(data)
        Y_train.append(label_map[file_name])
    if split == 'test':
        X_test.append(data)
        Y_test.append(label_map[file_name])
    if split == 'val':
        X_val.append(data)
        Y_val.append(label_map[file_name])

# X_train = np.array(X_train)
# Y_train = np.array(Y_train)
# X_val = np.array(X_val)
# Y_val = np.array(Y_val)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)

# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)
# print(X_val.shape)
# print(Y_val.shape)

batch_size = 8
# if num_classes > 10:
#     batch_size = 16

# prepare the dataloaders into a dict
train_dataloader = DataLoader(TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train)), batch_size=batch_size, shuffle=True)
del X_train
del Y_train
val_dataloader = DataLoader(TensorDataset(torch.Tensor(X_val),torch.Tensor(Y_val)), batch_size=batch_size, shuffle=True)
del X_val
del Y_val
test_dataloader = DataLoader(TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test)), batch_size=batch_size, shuffle=True)
del X_test
del Y_test
dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test':test_dataloader}

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}



save = True
model_path = './trained_model/spatio_tempo_model_{}_classes.pth'.format(num_classes)

# saves the time the process was started, to compute total time at the end
start = time.time()

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


# initalize the ResNet 18 version of this model
# model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=[2, 2, 2, 2]).to(device)
model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=[2, 2, 2, 2]).to(device)

# criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in paper sec 4.1
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs


criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # hyperparameters as given in paper sec 4.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs


epoch_resume = 0
num_epochs = 100

# # check if there was a previously saved checkpoint
if os.path.exists(model_path):
    # loads the checkpoint
    checkpoint = torch.load(model_path)
    print("Reloading from previously saved checkpoint")

    # restores the model and optimizer state_dicts
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_dict'])
    
    # obtains the epoch the training is to resume from
    epoch_resume = checkpoint["epoch"]


#training model
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
    # each epoch has a training and validation step, in that order
    for phase in ['train', 'val']:

        # reset the running loss and corrects
        running_loss = 0.0
        running_corrects = 0

        # set model to train() or eval() mode depending on whether it is trained
        # or being validated. Primarily affects layers such as BatchNorm or Dropout.
        if phase == 'train':
            # scheduler.step() is to be called once every epoch during training
            scheduler.step()
            model.train()
        else:
            model.eval()


        for inputs, labels in dataloaders[phase]:
            # move inputs and labels to the device the training is taking place on
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            # keep intermediate states iff backpropagation will be performed. If false, 
            # then all intermediate states will be thrown away during evaluation, to use
            # the least amount of memory possible.
            with torch.set_grad_enabled(phase=='train'):
                outputs = model(inputs)
                # we're interested in the indices on the max values, not the values themselves
                _, preds = torch.max(outputs, 1)  
                loss = criterion(outputs, labels)

                # Backpropagate and optimize iff in training mode, else there's no intermediate
                # values to backpropagate with and will throw an error.
                if phase == 'train':
                    loss.backward()
                    optimizer.step()   

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        if phase=='train':
            train_loss.append(epoch_loss)
            train_accuracy.append(epoch_acc)
        if phase=='val':
            val_loss.append(epoch_loss)
            val_accuracy.append(epoch_acc)

        print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

# save the model if save=True
    if save:
        torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': epoch_acc,
        'opt_dict': optimizer.state_dict(),
        }, model_path)
    # torch.save(model,model_path)

# print the total time needed, HH:MM:SS format
time_elapsed = time.time() - start    
print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60 :.4}s")

#printing and saving losses curve
epochs = range(1,num_epochs+1)
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./figures/train_val_losses_{}_class.png'.format(num_classes))
plt.show()


#testing model
model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=[2, 2, 2, 2])

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

# reset the running loss and corrects
running_loss = 0.0
running_corrects = 0

phase = 'test'

model.eval()

y_pred = []
y_true = []

criterion = nn.CrossEntropyLoss()

for inputs, labels in dataloaders[phase]:
    # move inputs and labels to the device the training is taking place on
    inputs = inputs
    labels = labels.type(torch.LongTensor)

    y_true.append(labels.tolist())
    

    outputs = model(inputs)
    

    _, preds = torch.max(outputs, 1)  
    
    loss = criterion(outputs, labels)

    y_pred.append(preds.tolist())

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

test_loss = running_loss / dataset_sizes[phase]
test_acc = running_corrects / dataset_sizes[phase]

print(f"{phase} Loss: {test_loss} Acc: {test_acc}")



