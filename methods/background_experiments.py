
from seek_code import SEEK
from data_handler import EleHandler
from classifier import Classifier
from backbone import Backbone
from concept_head import ConceptHead
from retrieval import Retrieval
from projector import Projector

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, SequentialSampler

import torch, math
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn.functional as F

print('CBM experiment')
dataset = EleHandler(subset='IDI_6')
train_indices, test_indices = dataset.split_perpendicular_to_elephants_and_encounters(split_sizes=[0.5,0.5], hour_delta = 0.2)

# Check if train and test indices intersect
train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=True)

full_dataset = EleHandler()
indices = np.random.permutation(len(full_dataset))

ch_train_loader = DataLoader(Subset(full_dataset, indices[:int(0.7 * len(indices))]), batch_size=512, shuffle=True)
ch_val_loader = DataLoader(Subset(full_dataset, indices[int(0.7 * len(indices)):int(0.85 * len(indices))]), batch_size=512, shuffle=True)
ch_test_loader = DataLoader(Subset(full_dataset, indices[int(0.85 * len(indices)):]), batch_size=512, shuffle=True)

code = 'baseline-3-MD'
print('===================', code, '===================')
r31 = Retrieval(experiment_code=code)

MD = Backbone(with_ears = True, lr = 1e-4, pretraining = "savannah_elephants", experiment_code=code)
ch1 = ConceptHead(experiment_code=code)
ch1.train(ch_train_loader, val_loader=ch_val_loader, backbone= MD, num_epochs=200)
ch1.test(ch_test_loader, backbone= MD)

ch1.test(test_loader, backbone= MD)
r31.evaluate_model(ch1, train_loader, test_loader, ba = MD, show_matches=False, show_tsne=False)

code = 'baseline-3-miew'
print('===================', code, '===================')
r32 = Retrieval(experiment_code=code)

miew = Backbone(with_ears = True, lr = 1e-4, model_name = "MiewID-msv3", experiment_code=code)
ch2 = ConceptHead(experiment_code=code)
ch2.train(ch_train_loader, val_loader=ch_val_loader, backbone= miew, num_epochs=200)
ch2.test(ch_test_loader, backbone= miew)

ch2.test(test_loader, backbone= miew)
r32.evaluate_model(ch2, train_loader, test_loader, ba = miew, show_matches=False, show_tsne=False)