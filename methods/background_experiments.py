from seek_code import SEEK
from data_handler import EleHandler
from seek_code import SEEK
from classifier import Classifier
#from backbone import Backbone
from concept_head import ConceptHead
from projector import Projector


from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import RandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

dataset = EleHandler(EFA_IDI_only=True)

train_indices, val_indices, test_indices = dataset.split_along_encounters([0.7,0.15,0.15])

train_loader, val_loader, test_loader = (DataLoader(dataset, batch_size=512, sampler=RandomSampler(indices), num_workers=4) for indices in [train_indices, val_indices, test_indices])

# Check the values of the labels
max_ele_label = 0
for batch in train_loader:
    max_ele_label = max(torch.max(batch[2]), max_ele_label)

print('Number of elephants:', max_ele_label.item() + 1)

code = 'ch+cl+pr'

print('Concepth head training')
ch = ConceptHead(print_every=1, expermient_code=code)
ch.train(train_loader, val_loader, num_epochs=2)
ch.test(test_loader)
print('Concepth frozen')
ch.freeze()

print('classifier training')
cl = Classifier(ch, num_classes=(max_ele_label + 1), expermient_code=code)
cl.train(train_loader, val_loader, num_epochs=3)
cl.test(test_loader) 

# Freeze the classifier layer
cl.freeze()
ch.freeze()


print('Projector Training with no intervention')

def intervene(predicted_concepts, true_concepts, correction_probability = 0.0):
    for i in range(predicted_concepts.shape[0]):
        if torch.rand(1).item() < correction_probability:
            predicted_concepts[i] = true_concepts[i]
    # To-do correction
    return predicted_concepts

pr = Projector(experiment_code=code) 
pr.train(train_loader=train_loader, val_loader=val_loader,  concept_head=ch, classifier=cl, num_epochs=300, intervention_fn = intervene)
pr.test(test_loader, cl, ch)


print('intervention at level 0.5')

def intervene(predicted_concepts, true_concepts, correction_probability = 0.5):
    for i in range(predicted_concepts.shape[0]):
        if torch.rand(1).item() < correction_probability:
            predicted_concepts[i] = true_concepts[i]
    # To-do correction
    return predicted_concepts

pr = Projector(experiment_code=code)

pr.train(train_loader, val_loader,  ch, cl, num_epochs=300, intervention_fn = intervene)
pr.test(test_loader, cl, ch)


print('intervention at level 1')

def intervene(predicted_concepts, true_concepts, correction_probability = 1):
    for i in range(predicted_concepts.shape[0]):
        if torch.rand(1).item() < correction_probability:
            predicted_concepts[i] = true_concepts[i]
    # To-do correction
    return predicted_concepts

pr = Projector(experiment_code=code)
pr.train(train_loader, val_loader,  ch, cl, num_epochs=300, intervention_fn = intervene)
pr.test(test_loader, cl, ch)