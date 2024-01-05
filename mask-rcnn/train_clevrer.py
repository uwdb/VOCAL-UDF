# -*- coding: utf-8 -*-
from dataset import *
from model import *
import torch

from detection.engine import train_one_epoch, evaluate
import detection.utils as utils

batch_size = 6
test_split = 1000 # how many images for testing set
num_epochs = 45

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 49

# use our dataset and defined transformations
dataset = CLEVRERDataset(get_transform(train=True))
dataset_test = CLEVRERDataset(get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-test_split])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_split:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
# and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    # lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    torch.save(model, os.path.join('output', 'models', 'mask-rcnn-clevrer_epoch-{}.pt'.format(epoch)))

print("training complete")

