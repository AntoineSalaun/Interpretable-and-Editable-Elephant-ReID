
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

train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

code = 'Projector_PoC'

def perfect_correction(hard_predicted_concepts, subject_SEEK, elephant_SEEK):
    return subject_SEEK

def oracle_correction(hard_predicted_concepts, subject_SEEK, elephant_SEEK):
    return elephant_SEEK

ch = ConceptHead(experiment_code=code)
ba = Backbone(pretraining='savannah_elephants', experiment_code=code)

r = Retrieval(experiment_code=code)

pr = Projector(criterion = 'ArcFace', lr = 0.001, scale = 64, margin = 0.5, experiment_code=code)

ba.freeze()
ch.freeze()
pr.unfreeze()

pr.train(train_loader, test_loader, ba, ch, num_epochs=200, intervention_fn=oracle_correction)

print('evaluation under oracle correction')
r.evaluate_model(pr, train_loader, test_loader, ba, ch, show_matches=False, show_tsne=False, show_plot=False, intervention_fn=oracle_correction)
print('evaluation under perfect subject correction')
r.evaluate_model(pr, train_loader, test_loader, ba, ch, show_matches=False, show_tsne=False, show_plot=False, intervention_fn=perfect_correction)
print('evaluation under no correction')
r.evaluate_model(pr, train_loader, test_loader, ba, ch, show_matches=False, show_tsne=False, show_plot=False, intervention_fn=None)