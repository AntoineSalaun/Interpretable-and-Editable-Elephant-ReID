import numpy as np
import os
from PIL import Image
# Make a dataloader that mixes wevrything randomly loads the images splits a train and test and computes the embeddings
from pathlib import Path
from data_handler import EleHandler
from seek_code import SEEK
from concept_head import ConceptHead
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn


dataset = EleHandler()
train_indices, val_indices, test_indices = dataset.split_along_encounters([0.7,0.15,0.15], identification='IDI')
train_loader, val_loader, test_loader = (DataLoader(dataset, batch_size=8, sampler=RandomSampler(indices), num_workers=0, pin_memory=True) for indices in [train_indices, val_indices, test_indices])

print('--------------------MSE - IDI only ---------------------')

ch_mse = ConceptHead(loss=nn.MSELoss(), print_every=1)
history_mse = ch_mse.train(train_loader, val_loader, num_epochs= 200)
_, _ = ch_mse.test(test_loader)

