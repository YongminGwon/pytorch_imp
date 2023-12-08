from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl

class NN(pl.LightningModule): # similar to nn.Module but has more additional functionality
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        loss, scores, y = self.commonstep(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self.commonstep(batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, scores, y = self.commonstep(batch, batch_idx)
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.lossfn(scores, y)
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters)
    
entire_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_ds, val_ds = random_split(entire_dataset, [50000, 10000])
test_ds = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True)

batch_size = 16
input_size = 28*28
num_classes = 10
device = torch.cuda()

train_loader = DataLoader(dataset = train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)


model = NN(input_size = input_size, num_classes=num_classes).to(device)

trainer = pl.Trainer()
trainer.fit(model, train_loader, val_loader)