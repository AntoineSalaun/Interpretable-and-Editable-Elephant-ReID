from seek_code import SEEK
from backbone import Backbone
from retrieval import Retrieval
from projector import Projector
from data_handler import EleHandler
from concept_head_tunneled import ConceptHeadTunneled, categorical_CE_loss, CrossedHeadNN, MultiHeadNN

from pathlib import Path
from torch.utils.data import DataLoader, Subset

code = 'DualRetrieval'

dataset = EleHandler(subset='IDI_6')
train_indices, test_indices = dataset.split_perpendicular_to_elephants_and_encounters(split_sizes=[0.5,0.5], hour_delta = 0.2)

train_subset = Subset(dataset, train_indices)
test_subset = Subset(dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)

MD_for_concepts.unfreeze()
concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code = code, layer = CrossedHeadNN(), reset_weights=False)
#concept_head.train(train_loader, test_loader, backbone = MD_for_concepts, num_epochs=50, predict_ele_SEEK = True)
concept_head.test(test_loader, backbone = MD_for_concepts, predict_ele_SEEK = True)

r = Retrieval(experiment_code=code)
r.evaluate_model(concept_head, ba= MD_for_concepts, train_loader=train_loader, test_loader=test_loader, show_matches=False, show_plot=False, show_tsne=False)

MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining="MD_finetuned_w", experiment_code=code)
r.evaluate_model(MD_finedtuned, train_loader=train_loader, test_loader=test_loader, show_matches=False, show_plot=False, show_tsne=False)

import torch
def correct_or_soft(concept_logits, subject_SEEK, elephant_SEEK):
        # Randomly choose between oracle correction and perfect correction
        if torch.rand(1) < 0.5:
            return elephant_SEEK
        else:
            return concept_logits

def correct_or_hard(concept_logits, subject_SEEK, elephant_SEEK):
        # Randomly choose between oracle correction and perfect correction
        if torch.rand(1) < 0.5:
            return elephant_SEEK
        else:
            return SEEK.closest_valid_one_hot(concept_logits)

def hard(concept_logits, subject_SEEK, elephant_SEEK):
    return SEEK.closest_valid_one_hot(concept_logits)

code = 'oracle-just-projector'

MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code)
pr = Projector(lr=5e-4, scale=64, margin=0.5, experiment_code=code, intervention_fn=SEEK.oracle_correction)
concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code = code, layer = CrossedHeadNN(), reset_weights=False)

# Freeze other models and unfreeze the projector
MD_for_concepts.freeze()
MD_finedtuned.freeze()
concept_head.freeze()
pr.unfreeze()

# Train the model
pr.train(
    train_loader, 
    test_loader, 
    backbone_for_concepts=MD_for_concepts, 
    backbone=MD_finedtuned, 
    concept_head=concept_head, 
    num_epochs=500
)

# Define intervention functions for testing
test_intervention_fns = {
    "100% Correction": SEEK.oracle_correction,
    "50% Correction + soft": correct_or_soft,
    "0% Correction": None,
    "0% Correction Hard": hard
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


code = 'oracle-backbone+projector'

MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code)
pr = Projector(lr=5e-4, scale=64, margin=0.5, experiment_code=code, intervention_fn=SEEK.oracle_correction)
concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code = code, layer = CrossedHeadNN(), reset_weights=False)

# Freeze other models and unfreeze the projector
MD_for_concepts.freeze()
MD_finedtuned.unfreeze()
concept_head.freeze()
pr.unfreeze()

# Train the model
pr.train(
    train_loader, 
    test_loader, 
    backbone_for_concepts=MD_for_concepts, 
    backbone=MD_finedtuned, 
    concept_head=concept_head, 
    num_epochs=500
)

# Define intervention functions for testing
test_intervention_fns = {
    "100% Correction": SEEK.oracle_correction,
    "50% Correction + soft": correct_or_soft,
    "0% Correction": None,
    "0% Correction Hard": hard
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