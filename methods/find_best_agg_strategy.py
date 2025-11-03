import optuna, torch
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from seek_code import SEEK
from backbone import Backbone
from retrieval import Retrieval
from projector import Projector
from data_handler import EleHandler
from concept_head_tunneled import ConceptHeadTunneled, categorical_CE_loss, CrossedHeadNN, MultiHeadNN

from pathlib import Path
from torch.utils.data import DataLoader, Subset
import os,json
import pandas as pd

# --- Start near your traditional rules ---------------------------------------
traditional_rules = {
    'sex': {'cutoff': 0.8, 'fraction': 2.0},
    'age': {'cutoff': 0.8, 'fraction': 1.0},
    'tusks': {'cutoff': 0.8, 'fraction': 1.0},
    'ear_most_prominent': {'cutoff': 0.9, 'fraction': 2.0},
    'ear_least_prominent': {'cutoff': 0.95, 'fraction': 2.5},
    'extremes': {'cutoff': 0.8, 'fraction': 1.5},
}

# Wide, permissive bounds for the optimizer
_BOUNDS = {
    k: {
        "cutoff": (0.2, 1.0),
        "fraction": (0.5, 3.5)  # search on log-scale
    } for k in traditional_rules
}

# --- One-time setup (queries fixed) ------------------------------------------
def setup(dictonary_path):
    dataset = EleHandler(subset='IDI_6', dictonary_path=dictonary_path)
    tr_idx, te_idx = dataset.split_perpendicular_to_elephants_and_encounters([0.5, 0.5], hour_delta=0.2)
    train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=64, shuffle=True, num_workers=4)
    test_loader  = DataLoader(Subset(dataset, te_idx),  batch_size=64, shuffle=False, num_workers=4)

    code = 'rule_opt_minimal'
    r = Retrieval()
    MD = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
    head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code=code,
                               layer=CrossedHeadNN(), reset_weights=False, pretraining='concept_w')

    with torch.no_grad():
        q_emb, q_lab = head.collect_embeddings(test_loader, backbone=MD,
                                               intervention_fn=SEEK.perfect_correction,
                                               aggregate_seeks=False)
    return dict(dataset=dataset, train_loader=train_loader, retrieval=r, MD=MD, head=head,
                q_emb=q_emb, q_lab=q_lab)

PIPE = setup("/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_optimized.csv")

# --- Minimal objective: maximize Recall@1 ------------------------------------
def _suggest_rules(trial):
    rules = {}
    for f, b in _BOUNDS.items():
        rules[f] = {
            "cutoff":   trial.suggest_float(f"{f}.cutoff", *b["cutoff"]),
            "fraction": trial.suggest_float(f"{f}.fraction", *b["fraction"], log=True),
        }
    return rules

def objective(trial):
    rules = _suggest_rules(trial)
    with torch.no_grad():
        gal_emb, gal_lab = PIPE["head"].collect_embeddings(
            PIPE["train_loader"], backbone=PIPE["MD"],
            intervention_fn=SEEK.perfect_correction,
            aggregate_seeks=True, aggregate_rules=rules
        )
        sim = PIPE["retrieval"].similarity_matrix(PIPE["q_emb"], gal_emb, distance='seek_homemade_3')
        r1 = PIPE["retrieval"].compute_recall_at_k(similarity_matrix=sim,
                                                   query_labels=PIPE["q_lab"],
                                                   gallery_labels=gal_lab, k=1)
    return float(r1) if r1 is not None else 0.0

def tune_rules_for_recall1(n_trials=100, seed=0):
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed, n_startup_trials=min(20, n_trials//5)))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {f: {"cutoff": study.best_params[f"{f}.cutoff"],
                "fraction": study.best_params[f"{f}.fraction"]} for f in _BOUNDS}
    return best, study.best_value

# --- Run ----------------------------------------------------------------------
best_rules, best_r1 = tune_rules_for_recall1(n_trials=2500, seed=42)
print("Best Recall@1:", best_r1)
print("Best rules:")
for k, v in best_rules.items(): print(k, v)
