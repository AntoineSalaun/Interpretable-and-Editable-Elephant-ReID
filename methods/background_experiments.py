from seek_code import SEEK
from data_handler import EleHandler
from classifier import Classifier
from backbone import Backbone
from concept_head import ConceptHead
from projector import Projector

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import RandomSampler

import torch, math
import torch.nn as nn
import torch.optim as optim
import pandas as pd


dataset = EleHandler(subset='EFA_IDI')

train_indices, val_indices, test_indices = dataset.split_along_encounters([0.7,0.15,0.15])

train_loader, val_loader, test_loader = (DataLoader(dataset, batch_size=128, sampler=RandomSampler(indices), num_workers=4) for indices in [train_indices, val_indices, test_indices])

# Check the values of the labels
max_ele_label = 0
for batch in train_loader:
    max_ele_label = max(torch.max(batch[2]), max_ele_label)

print('Number of elephants:', max_ele_label.item() + 1)

num_classes = max_ele_label.item() + 1
code = 'Elephant_Classifier'


print('===============stage1====================')
ba = Backbone(pretraining = "savannah_elephants")
ba.freeze()

ch = ConceptHead(print_every=1, experiment_code=code)
#ch.train(train_loader, val_loader, ba, num_epochs=30)
ch.test(test_loader, ba)

ch.freeze()

print('===============stage2====================')
cl = Classifier( num_classes=num_classes, experiment_code=code)
cl.train(train_loader, val_loader, backbone = ba, num_epochs=200)
cl.test(test_loader, backbone = ba) 

# Freeze the classifier layer
cl.freeze()

print('===============stage3====================')
def no_intervention(predicted_concepts, true_concepts, correction_probability = 0.0):
    return predicted_concepts

def intervene(predicted_concepts, true_concepts, correction_probability = 0.5):
    for i in range(predicted_concepts.shape[0]):
        if torch.rand(1).item() < correction_probability:
            predicted_concepts[i] = true_concepts[i]
    return predicted_concepts

def perfect_correction(predicted_concepts, true_concepts, correction_probability = 1.0):
    return true_concepts

pr = Projector(experiment_code=code) 

ba.freeze() # freeze the backbone
ch.freeze() # freeze the concept head
cl.freeze() # freeze the classifier
pr.unfreeze() # unfreeze the projector

pr.train(train_loader, val_loader, ba,  ch, cl, num_epochs=100, intervention_fn=no_intervention)
pr.test(test_loader, ba, cl, ch)
pr.test(test_loader, ba, cl, ch, intervention_fn=intervene)
pr.test(test_loader, ba, cl, ch, intervention_fn=perfect_correction)

print('===============stage4====================')
pr.unfreeze()
cl.unfreeze()
ch.freeze()
ba.freeze()

pr.train(train_loader, val_loader, ba, ch, cl, num_epochs=100, intervention_fn = intervene, reset_wegihts=False) 
pr.test(test_loader, ba, cl, ch, intervention_fn = no_intervention)
pr.test(test_loader, ba, cl, ch, intervention_fn = intervene)
pr.test(test_loader, ba, cl, ch, intervention_fn = perfect_correction)

print('===============stage5====================')
import torch
import math
import torch.nn.functional as F

class ArcFaceWithMargin(torch.nn.Module):
    def __init__(self, num_classes, s=30.0, margin=0.5):
        """
        ArcFace with Angular Margin, with added numerical stability.
        Args:
            num_classes: Number of classes in the classification task.
            s: Scale factor to amplify logits for softmax.
            margin: Angular margin (in radians) to enhance class separability.
        """
        super(ArcFaceWithMargin, self).__init__()
        self.s = s  # Scale factor
        self.margin = margin  # Angular margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits: Raw logits from the linear layer (batch_size, num_classes).
            labels: Ground truth labels (batch_size,).
        Returns:
            loss: Cross-Entropy Loss with Angular Margin.
        """
        # Normalize the logits
        norm = torch.norm(logits, dim=1, keepdim=True) + 1e-8
        logits_norm = logits / norm

        # Create one-hot encoding for the labels
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1.0)

        # Compute cos(theta) for the target classes
        cos_theta = logits_norm.gather(1, labels.view(-1, 1)).squeeze(1)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # Tighter clamping for stability

        # Calculate sin(theta)
        sin_theta_squared = 1.0 - cos_theta**2
        sin_theta_squared = sin_theta_squared.clamp(0, 1)  # Ensure no negative values
        sin_theta = torch.sqrt(sin_theta_squared)

        # Calculate cos(theta + margin)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Ensure numerical stability for the margin update
        cos_theta_m = cos_theta_m.clamp(-1, 1)

        # Update logits with the margin
        logits_with_margin = logits_norm.clone()
        logits_with_margin.scatter_(1, labels.view(-1, 1), cos_theta_m.unsqueeze(1))

        # Scale the logits
        scaled_logits = logits_with_margin * self.s

        # Compute Cross-Entropy Loss
        loss = F.cross_entropy(scaled_logits, labels)
        print('loss', loss)
        return loss

ba.freeze()
ch.freeze()
pr.unfreeze()
cl.unfreeze()

pr.loss_fn = ArcFaceWithMargin(num_classes)

pr.train(train_loader, val_loader, ba, ch, cl, num_epochs=100, intervention_fn = no_intervention, reset_wegihts=False) 
pr.test(test_loader, ba, cl, ch, intervention_fn = no_intervention)
pr.test(test_loader, ba, cl, ch, intervention_fn = intervene)
pr.test(test_loader, ba, cl, ch, intervention_fn = perfect_correction)


