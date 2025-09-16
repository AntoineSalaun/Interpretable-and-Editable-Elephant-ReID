
from seek_code import SEEK
from backbone import Backbone
from retrieval import Retrieval
from projector import Projector
from data_handler import EleHandler
from concept_head_tunneled import ConceptHeadTunneled, categorical_CE_loss, CrossedHeadNN, MultiHeadNN

from pathlib import Path
from torch.utils.data import DataLoader, Subset


dataset = EleHandler(subset='IDI_6')
train_indices, test_indices = dataset.split_perpendicular_to_elephants_and_encounters(split_sizes=[0.5,0.5], hour_delta = 0.2)

train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=True)

code = 'md-only-alpha=0-lr=5e-6'

MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code)
pr = Projector(loss_type ='ArcFace', lr=5e-6, scale=64, margin=0.5, experiment_code=code, intervention_fn=SEEK.correct_or_soft, alpha = 0)

import torch.nn as nn
pr.layer = nn.Sequential(
            nn.Linear(63, 2304),
            nn.LeakyReLU()
        ).to('cuda')

concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code = code, layer = CrossedHeadNN(), reset_weights=False)


# Freeze other models and unfreeze the projector
MD_for_concepts.freeze()
MD_finedtuned.unfreeze()
concept_head.freeze()
pr.freeze()

# Train the model
pr.train(
    train_loader, 
    test_loader, 
    backbone_for_concepts=MD_for_concepts, 
    backbone=MD_finedtuned, 
    concept_head=concept_head, 
    num_epochs=800
)

# Define intervention functions for testing
test_intervention_fns = {
    "100% Correction": SEEK.oracle_correction,
    "50% Correction + soft": SEEK.correct_or_soft,
    "0% Correction": None,
    "0% Correction Hard": SEEK.hard
}

# Run the structured test method
pr.test(
    backbone_for_concepts=MD_for_concepts, 
    backbone=MD_finedtuned, 
    concept_head=concept_head, 
    train_loader=train_loader, 
    test_loader=test_loader, 
    intervention_fns=test_intervention_fns
)