import os
import numpy as np

from PIL import Image

# Make a dataloader that mixes wevrything randomly loads the images splits a train and test and computes the embeddings
from pathlib import Path

from data_handler import EleHandler
from seek_code import SEEK
from concept_head import ConceptHead

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import torch.nn as nn

print('working from : ', os.getcwd())
# Create dataset and dataloaders
dataset = EleHandler()

# I cut along the encounters to make sure that the same encounter is not in both train and test (with the criteria that two images taken less than 2 hours appart are in the same encounter)
train_indices, val_indices, test_indices = dataset.split_along_encounters([0.7,0.15,0.15], )

train_loader, val_loader, test_loader = (DataLoader(dataset, batch_size=8, sampler=RandomSampler(indices), num_workers=0) for indices in [train_indices, val_indices, test_indices])

print('--------------------MSE---------------------')
# cross-entropy loss
ch_mse = ConceptHead(loss=nn.MSELoss(), print_every=1)
history_mse = ch_mse.train(train_loader, val_loader, num_epochs= 200)
_, _ = ch_mse.test(test_loader)

print('--------------------CE---------------------')

ch = ConceptHead(print_every=1)
ch.train(train_loader, val_loader, num_epochs= 50)
_,_ = ch.test(test_loader)

print('--------------------homemade loss---------------------')


def split_outputs(prob_vector):
    """
    Converts a probability one-hot predicted tensor into a list of smaller tensors, each representing an attribute
    """
    val_prob_list = []
    index = 0

    for att in SEEK.attribute_names:
        slice_length = SEEK.lengths[att]
        prob_slice = prob_vector[:, index:index + slice_length]
        val_prob_list.append(prob_slice)
        
        index += slice_length

    return val_prob_list


def character_regularized_loss(outputs, labels, main_loss_fn = nn.MSELoss()):

    """
    Computes the loss of the model such that each attribute of the SEEK code is assessed equally, regardless of the number of values it can take (and of the size of its one-hot econding)
    """
    outputs_list = split_outputs(outputs)
    labels_list = split_outputs(labels)

    loss = 0
    for i in range(len(outputs_list)):
        loss += main_loss_fn(outputs_list[i], labels_list[i])/SEEK.lengths[SEEK.attribute_names[i]]

    return loss

# cross-entropy loss
ch_homemade = ConceptHead(loss=character_regularized_loss, print_every=1)
history_homemade = ch_homemade.train(train_loader, val_loader, num_epochs= 200)
_, _ = ch_homemade.test(test_loader)



