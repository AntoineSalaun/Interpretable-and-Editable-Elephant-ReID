# import the library
import os
from pathlib import Path
from random import random

import wandb

from seek_code import SEEK
from backbone import Backbone
from MegaDescriptor import MegaDescriptor
from miewid import MiewID
from retrieval import Retrieval
from projector import Projector
from elephantbook_baseline import ElephantBookBaseline
from data_handler import EleHandler
from concept_head_tunneled import ConceptHeadTunneled, categorical_CE_loss, CrossedHeadNN, MultiHeadNN, ThreeHeadNN
from torch.utils.data import DataLoader, Subset
import argparse

# Implement Argparse for the configuration
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
parser.add_argument('--dataset', type=str, default='mara', help='Dataset to use. Experiments default to mara.')
parser.add_argument('--subset', type=str, default='2+encounters', help='Subset of the dataset to use')
parser.add_argument('--pipeline_name', type=str, default=None, help='Pipeline to use: PretrainMegaDescriptor, MiewID, FinetuningTripleCropBackbone, Retrieval_on_SEEK_codes, ElephantBook, TrainingConceptHead, CHAIR')

parser.add_argument('--backbone_pretraining', type=str, default='backbone_w_gold', help='Pretraining for the backbone: a single checkpoint name/path, or a comma-separated complete,left,right triple')
parser.add_argument('--backbone_lr', type=float, default=5e-6, help='Learning rate for the backbone')
parser.add_argument('--backbone_wd', type=float, default=0.0, help='Weight decay for the backbone')
parser.add_argument('--backbone_layer_norm', type=lambda x: str(x).lower() == 'true', default=False, help='Whether to use layer normalization in the backbone')
parser.add_argument('--mega_descriptor_mode', type=str, default='finetuning', help='Mode for PretrainMegaDescriptor: finetuning, zero_shot, savannah_elephants')
parser.add_argument('--mega_descriptor_crop', type=str, default='complete', help='Crop to use for PretrainMegaDescriptor: complete, left_ear, right_ear')
parser.add_argument('--miewid_mode', type=str, default='out_of_box', help='Mode for MiewID: out_of_box or finetuning')

parser.add_argument('--backbone_for_concepts_pretraining', type=str, default='backbone_for_concepts_gold', help='Pretraining for the backbone used for concept head: a single checkpoint name/path, or a comma-separated complete,left,right triple')
parser.add_argument('--backbone_for_concepts_lr', type=float, default=5e-6, help='Learning rate for the backbone used for concept head')

parser.add_argument('--concept_head_pretraining', type=str, default='concept_w_gold', help='Pretraining for the concept head: a single checkpoint name/path')
parser.add_argument('--concept_head_lr', type=float, default=1e-5, help='Learning rate for the concept head')
parser.add_argument('--concept_head_wd', type=float, default=None, help='Weight decay for the concept head')
parser.add_argument('--concept_head_architecture', type=str, default='three', help='Concept head architecture: crossed, multi, or three')
parser.add_argument('--concept_head_target_is_ele_seek', type=lambda x: str(x).lower() == 'true', default=False, help='Whether the target for the concept head is the elephant ID')
parser.add_argument('--concept_head_training', type=str, default='finetuning', help='Type of training for the Concept Head')
parser.add_argument('--concept_distance', type=str, default='seek_homemade_3', help='Distance function to use for concept head')

#parser.add_argument('--projector_pretraining', type=str, default=None, help='Pretraining for the projector')
parser.add_argument('--correction_at_training', type=str, default=None, help='Correction policy at projector training time. Bare numbers mean sighting-level correction.')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for the projector')
parser.add_argument('--fusion_method', type=str, default='summing')
parser.add_argument('--projector_training_policy', type=str, default='joint', help='Projector training policy: joint, freeze_backbone, sequential, or seq')
parser.add_argument('--projector_architecture', type= str, default = 'small')
parser.add_argument('--projector_lr', type=float, default=1e-3, help='Learning rate for the projector')
parser.add_argument('--projector_wd', type=float, default=1e-5, help='Weight decay for the projector')
parser.add_argument('--layer_norm', type=lambda x: str(x).lower() == 'true', default=False, help='Whether to use layer normalization')
parser.add_argument('--alpha_learnable', type=lambda x: str(x).lower() == 'true', default=False, help='Whether alpha is learnable')
parser.add_argument('--test_concept_head_before_projector', type=lambda x: str(x).lower() == 'true', default=False, help='Whether to test the concept head once before projector training')

parser.add_argument('--gallery_correction_fn', type=str, default=None, help='Correction policy for gallery SEEK codes at test time. Bare numbers mean sighting-level correction.')
parser.add_argument('--query_correction_fn', type=str, default=None, help='Correction policy for query SEEK codes at test time. Bare numbers mean image-level correction; use sighting_p for sighting-level correction.')
parser.add_argument('--elephantbook_backbone_weight', type=float, default=0.5, help='Weight on backbone confidence in the ElephantBook-style linear fusion.')
parser.add_argument('--elephantbook_wildcard_distance', type=float, default=0.6, help='SEEK attribute distance used when either value is a wildcard in the ElephantBook baseline.')
parser.add_argument('--elephantbook_query_chunk_size', type=int, default=512, help='Number of query embeddings scored per chunk in the ElephantBook baseline.')
parser.add_argument('--elephantbook_visual_with_ears', type=lambda x: str(x).lower() == 'true', default=True, help='Whether the ElephantBook visual backbone uses ear crops in addition to the body crop.')

parser.add_argument('--print_every', type=int, default=5, help='How often to print during training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
parser.add_argument('--wandb_name', type=str, default=None, help='Name of the wandb run')
parser.add_argument('--wandb_group', type=str, default=None, help='Group of the wandb run')
parser.add_argument('--save_best', type=lambda x: str(x).lower() == 'true', default=True, help='Whether to save the best on validation model during training')
parser.add_argument('--save_embeddings', type=lambda x: str(x).lower() == 'true', default=False, help='Whether to save the embeddings of the best model during training')

args = parser.parse_args()


# ---- 3) Build your CONFIG with real callables
CONFIG = {
    "pipeline_name": args.pipeline_name,
    "epochs": args.epochs,
    "dataset": args.dataset,
    "subset": args.subset,

    "backbone_pretraining": args.backbone_pretraining,
    "backbone_lr": args.backbone_lr,
    "backbone_wd": args.backbone_wd,
    "backbone_layer_norm": args.backbone_layer_norm,
    "mega_descriptor_mode": args.mega_descriptor_mode,
    "mega_descriptor_crop": args.mega_descriptor_crop,
    "miewid_mode": args.miewid_mode,

    "backbone_for_concepts_pretraining": args.backbone_for_concepts_pretraining,
    "backbone_for_concepts_lr": args.backbone_for_concepts_lr,

    "concept_head_pretraining": args.concept_head_pretraining,
    "concept_head_lr": args.concept_head_lr,
    "concept_head_wd": args.concept_head_wd,
    "concept_head_architecture": args.concept_head_architecture,
    "concept_head_target_is_ele_seek": args.concept_head_target_is_ele_seek,
    "ConceptHead_training": args.concept_head_training,
    "concept_distance": args.concept_distance,

    #"projector_pretraining": args.projector_pretraining,
    "alpha": args.alpha,
    "fusion_method": args.fusion_method,
    "projector_training_policy": args.projector_training_policy,
    "projector_architecture": args.projector_architecture,
    "projector_lr": args.projector_lr,
    "projector_wd": args.projector_wd,
    "layer_norm": args.layer_norm,
    "alpha_learnable": args.alpha_learnable,
    "test_concept_head_before_projector": args.test_concept_head_before_projector,

    "correction_at_training": args.correction_at_training,
    "gallery_correction_fn": args.gallery_correction_fn,
    "query_correction_fn": args.query_correction_fn,
    "elephantbook_backbone_weight": args.elephantbook_backbone_weight,
    "elephantbook_wildcard_distance": args.elephantbook_wildcard_distance,
    "elephantbook_query_chunk_size": args.elephantbook_query_chunk_size,
    "elephantbook_visual_with_ears": args.elephantbook_visual_with_ears,

    "print_every": args.print_every,
    "batch_size": args.batch_size,
    "wandb_name": args.wandb_name,
    "wandb_group": args.wandb_group,
    "save_best": args.save_best,
    "save_embeddings": args.save_embeddings
}


# Resolve repository root so relative assets work regardless of CWD
REPO_ROOT = Path(__file__).resolve().parent.parent


def build_concept_head_layer(architecture_name):
    architecture_name = str(architecture_name).lower()

    if architecture_name in {'crossed', 'crossedheadnn'}:
        return CrossedHeadNN()
    if architecture_name in {'multi', 'multiheadnn'}:
        return MultiHeadNN()
    if architecture_name in {'three', 'threeheadnn'}:
        return ThreeHeadNN()

    raise ValueError(
        f"Unsupported concept_head_architecture: {architecture_name} "
        "(expected 'crossed', 'multi', or 'three')"
    )


def correction_summary(fn):
    return {
        "name": getattr(fn, "__name__", str(fn)),
        "scope": getattr(fn, "correction_scope", None),
        "probability": getattr(fn, "correction_probability", None),
        "selected_units": len(getattr(fn, "corrected_units", []) or []),
    }

# start a new experiment
with wandb.init(project="CBM-ReID", name=args.wandb_name, group=args.wandb_group, config=CONFIG) as run:

    cfg = run.config

    if str(cfg.dataset).lower() == 'mara':
        dataset = EleHandler(subset=cfg.subset, dataset_type='mara')
        
        # Now the indices are hard-coded and work with 
        #train_indices, test_indices = dataset.split_parallel_to_encounters()
        weights_dir = REPO_ROOT / 'weights'
        train_indices_path = weights_dir / 'train_indices_gold.txt'
        test_indices_path = weights_dir / 'test_indicies_gold.txt'
        
        if not train_indices_path.exists() or not test_indices_path.exists():
            raise FileNotFoundError(f"Indices files not found. Please check:\n  - {train_indices_path}\n  - {test_indices_path}")
        
        with train_indices_path.open('r') as f:
            train_indices = [int(line.strip()) for line in f.readlines()]
        with test_indices_path.open('r') as f:
            test_indices = [int(line.strip()) for line in f.readlines()]

        wandb.summary.update({"data in train": len(train_indices),"data in test": len(test_indices)})
        wandb.summary.update({"train_indices": train_indices, "test_indices": test_indices})
        
        import torch  # Ensure torch is imported at the top of the file

        torch.manual_seed(42)  # Set the seed for reproducibility
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    
    if cfg.pipeline_name == 'PretrainMegaDescriptor':
        if cfg.mega_descriptor_mode == 'finetuning':
            pretraining = cfg.backbone_pretraining
        elif cfg.mega_descriptor_mode == 'zero_shot':
            pretraining = 'zero_shot'
        elif cfg.mega_descriptor_mode == 'savannah_elephants':
            pretraining = 'savannah_elephants'
        else:
            raise ValueError(
                f"Unsupported mega_descriptor_mode: {cfg.mega_descriptor_mode} "
                "(expected 'finetuning', 'zero_shot' or 'savannah_elephants')"
            )

        model = MegaDescriptor(
            pretraining=pretraining,
            lr=cfg.backbone_lr,
            wd=cfg.backbone_wd,
            experiment_code=wandb.run.name,
            print_every=cfg.print_every,
            num_classes=len(dataset.ele_id_to_label),
            crop_type=cfg.mega_descriptor_crop,
        )

        if cfg.mega_descriptor_mode == 'finetuning':
            model.train(train_loader, test_loader, num_epochs=cfg.epochs, save_best=cfg.save_best)
        model.test(train_loader, test_loader)

    elif cfg.pipeline_name == 'MiewID':
        if cfg.miewid_mode == 'finetuning':
            pretraining = cfg.backbone_pretraining
        elif cfg.miewid_mode in {'out_of_box', 'zero_shot'}:
            pretraining = 'out_of_box'
        else:
            raise ValueError(
                f"Unsupported miewid_mode: {cfg.miewid_mode} "
                "(expected 'out_of_box' or 'finetuning')"
            )

        model = MiewID(
            pretraining=pretraining,
            lr=cfg.backbone_lr,
            wd=cfg.backbone_wd,
            experiment_code=wandb.run.name,
            print_every=cfg.print_every,
            num_classes=len(dataset.ele_id_to_label),
        )
        if cfg.miewid_mode == 'finetuning':
            model.train(train_loader, test_loader, num_epochs=cfg.epochs, save_best=cfg.save_best)
        model.test(train_loader, test_loader)

    elif cfg.pipeline_name == 'FinetuningTripleCropBackbone':
        model = Backbone(
            model_name="MegaDescriptor",
            with_ears=True,
            pretraining=cfg.backbone_pretraining,
            lr=cfg.backbone_lr,
            wd=cfg.backbone_wd,
            experiment_code=wandb.run.name,
            print_every=cfg.print_every,
            num_classes=len(dataset.ele_id_to_label),
            layer_norm=cfg.backbone_layer_norm,
        )

        model.train(train_loader, test_loader, num_epochs=cfg.epochs, save_best=cfg.save_best)
        model.test(train_loader, test_loader)
        
    elif cfg.pipeline_name == 'Retrieval_on_SEEK_codes':
        r = Retrieval()

        MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining=cfg.backbone_for_concepts_pretraining, experiment_code=wandb.run.name)
        concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, wd=cfg.concept_head_wd, experiment_code=wandb.run.name, layer=build_concept_head_layer(cfg.concept_head_architecture), reset_weights=False, pretraining=cfg.concept_head_pretraining)

        # Collect embeddings and compute similarity matrix
        gallery_correction_fn = SEEK.get_correction_policy(cfg.gallery_correction_fn, dataset, default_scope="sighting")
        query_correction_fn = SEEK.get_correction_policy(cfg.query_correction_fn, dataset, default_scope="image")
        wandb.summary.update({
            "gallery_correction_policy": correction_summary(gallery_correction_fn),
            "query_correction_policy": correction_summary(query_correction_fn),
        })
            
        gallery_embeddings, gallery_labels = concept_head.collect_embeddings(train_loader, backbone=MD_for_concepts, intervention_fn=gallery_correction_fn)
        query_embeddings, query_labels = concept_head.collect_embeddings(test_loader, backbone=MD_for_concepts, intervention_fn=query_correction_fn)

        test_similarity_matrix = r.similarity_matrix(query_embeddings, gallery_embeddings, distance=cfg.concept_distance)

        # Compute and print recalls
        test_recalls = {k: r.compute_recall_at_k(similarity_matrix=test_similarity_matrix, query_labels=query_labels, gallery_labels=gallery_labels, k=k) for k in [1, 5, 10, 20, 100]}
        wandb.summary.update({f"Test-Recall@{k}": test_recalls[k] * 100 for k in [1, 5, 10, 20, 100]})

    elif cfg.pipeline_name == 'ElephantBook':
        MD_for_concepts = Backbone(
            model_name="MegaDescriptor",
            pretraining=cfg.backbone_for_concepts_pretraining,
            experiment_code=wandb.run.name,
        )
        MD_finetuned = Backbone(
            model_name="MegaDescriptor",
            pretraining=cfg.backbone_pretraining,
            experiment_code=wandb.run.name,
            num_classes=len(dataset.ele_id_to_label),
            layer_norm=cfg.backbone_layer_norm,
            with_ears=cfg.elephantbook_visual_with_ears,
        )
        concept_head = ConceptHeadTunneled(
            loss=categorical_CE_loss,
            wd=cfg.concept_head_wd,
            experiment_code=wandb.run.name,
            layer=build_concept_head_layer(cfg.concept_head_architecture),
            reset_weights=False,
            pretraining=cfg.concept_head_pretraining,
        )

        gallery_correction_fn = SEEK.get_correction_policy(cfg.gallery_correction_fn, dataset, default_scope="sighting")
        query_correction_fn = SEEK.get_correction_policy(cfg.query_correction_fn, dataset, default_scope="image")
        wandb.summary.update({
            "gallery_correction_policy": correction_summary(gallery_correction_fn),
            "query_correction_policy": correction_summary(query_correction_fn),
        })

        baseline = ElephantBookBaseline(
            backbone_weight=cfg.elephantbook_backbone_weight,
            wildcard_distance=cfg.elephantbook_wildcard_distance,
            query_chunk_size=cfg.elephantbook_query_chunk_size,
        )
        recalls, average_seek_distance = baseline.evaluate(
            backbone=MD_finetuned,
            backbone_for_concepts=MD_for_concepts,
            concept_head=concept_head,
            train_loader=train_loader,
            test_loader=test_loader,
            gallery_correction_fn=gallery_correction_fn,
            query_correction_fn=query_correction_fn,
        )

        ks = [1, 5, 10, 20, 100]
        print(
            "ElephantBook combined - "
            + ", ".join([f"Recall@{k}: {recalls['combined'][k] * 100:.2f}%" for k in ks])
        )
        wandb.summary.update({f"Test-Recall@{k}": recalls["combined"][k] * 100 for k in ks})
        wandb.summary.update({f"ElephantBook/Combined Recall@{k}": recalls["combined"][k] * 100 for k in ks})
        wandb.summary.update({f"ElephantBook/Backbone Recall@{k}": recalls["backbone"][k] * 100 for k in ks})
        wandb.summary.update({f"ElephantBook/SEEK Recall@{k}": recalls["seek"][k] * 100 for k in ks})
        wandb.summary.update({
            "ElephantBook/backbone_weight": cfg.elephantbook_backbone_weight,
            "ElephantBook/seek_weight": 1.0 - cfg.elephantbook_backbone_weight,
            "ElephantBook/wildcard_distance": cfg.elephantbook_wildcard_distance,
            "ElephantBook/average_pairwise_seek_distance": average_seek_distance,
            "ElephantBook/visual_with_ears": cfg.elephantbook_visual_with_ears,
        })

    elif cfg.pipeline_name == 'TrainingConceptHead':

        backbone_for_concepts = Backbone(model_name="MegaDescriptor", pretraining=args.backbone_for_concepts_pretraining, experiment_code=wandb.run.name, num_classes=len(dataset.ele_id_to_label))
        
        if cfg.ConceptHead_training == 'probing': backbone_for_concepts.freeze()
        elif cfg.ConceptHead_training == 'finetuning': backbone_for_concepts.unfreeze()
        else: raise ValueError(f"Unsupported ConceptHead training type: {cfg.ConceptHead_training} (expected 'probing' or 'finetuning')")

        if cfg.concept_head_target_is_ele_seek is not True and cfg.concept_head_target_is_ele_seek is not False:
            raise ValueError(f"concept_head_target_is_ele_seek must be a boolean (True or False), got ambiguous {cfg.concept_head_target_is_ele_seek}")
        
        reset_weights = cfg.concept_head_pretraining is None or str(cfg.concept_head_pretraining).lower() == 'none'  # If no pretraining, reset weights of the Concept Head
        
        CH = ConceptHeadTunneled(lr=args.concept_head_lr, wd=cfg.concept_head_wd, loss=categorical_CE_loss, experiment_code=wandb.run.name, layer=build_concept_head_layer(cfg.concept_head_architecture), reset_weights=reset_weights, pretraining=cfg.concept_head_pretraining)

        #Because I suspect that CH has access to some pretraining, I test it before running
        print("Testing Concept Head before training:")
        CH.test(test_loader, backbone=backbone_for_concepts, predict_ele_SEEK=cfg.concept_head_target_is_ele_seek)

        CH.train(train_loader, test_loader, backbone=backbone_for_concepts, num_epochs=args.epochs, predict_ele_SEEK=cfg.concept_head_target_is_ele_seek, save_best = cfg.save_best)
        CH.test(test_loader, backbone=backbone_for_concepts, predict_ele_SEEK=cfg.concept_head_target_is_ele_seek)

    elif cfg.pipeline_name == 'CHAIR': #--backbone_for_concepts_pretraining backbone_for_concepts_w_e213_finetuned_subject_high_lr --backbone_pretraining MD_finetuned_mara --concept_head_pretraining concept_head_w_e213_finetuned_subject_high_lr  --concept_head_target_is_ele_seek False --correction_at_training perfect_correction

        MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining=cfg.backbone_for_concepts_pretraining, experiment_code=wandb.run.name, num_classes=len(dataset.ele_id_to_label), lr=cfg.backbone_for_concepts_lr, print_every=cfg.print_every )
        MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining=args.backbone_pretraining, experiment_code=wandb.run.name, num_classes=len(dataset.ele_id_to_label), lr=cfg.backbone_lr, print_every=cfg.print_every)
        
        concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, wd=cfg.concept_head_wd, experiment_code = wandb.run.name, layer = build_concept_head_layer(cfg.concept_head_architecture), reset_weights=False, pretraining=cfg.concept_head_pretraining)

        if cfg.test_concept_head_before_projector:
            print("Testing Concept Head before projector training:")
            concept_head.test(test_loader, backbone=MD_for_concepts, predict_ele_SEEK=cfg.concept_head_target_is_ele_seek)
        
        # Get the correction policy using the unified helper
        intervention_fn = SEEK.get_correction_policy(cfg.correction_at_training, dataset, default_scope="sighting")
        wandb.summary.update({"training_correction_policy": correction_summary(intervention_fn)})

        pr = Projector(loss_type ='ArcFace', lr=cfg.projector_lr, scale=64, margin=0.5, experiment_code=wandb.run.name, intervention_fn=intervention_fn, alpha = cfg.alpha, reset_weights = True, network_type=cfg.projector_architecture, num_classes=len(dataset.ele_id_to_label), wd = cfg.projector_wd, layer_norm=cfg.layer_norm, alpha_learnable = cfg.alpha_learnable)

        # Freeze other models and unfreeze the projector
        MD_for_concepts.freeze()
        concept_head.freeze()

        projector_training_policy = str(cfg.projector_training_policy).lower()
        if projector_training_policy == 'joint':
            MD_finedtuned.unfreeze()
        elif projector_training_policy in {'freeze_backbone', 'sequential', 'seq'}:
            MD_finedtuned.freeze()
        else:
            raise ValueError(
                f"Unsupported projector_training_policy: {cfg.projector_training_policy} "
                "(expected 'joint', 'freeze_backbone', 'sequential', or 'seq')"
            )
        pr.unfreeze()

        # Train the model
        pr.train(
            train_loader, 
            test_loader, 
            backbone_for_concepts=MD_for_concepts, 
            backbone=MD_finedtuned, 
            concept_head=concept_head, 
            num_epochs=args.epochs,
            save_best = cfg.save_best,
            save_embeddings=cfg.save_embeddings)
        
        # Add percentage-based correction policies using get_correction_policy
        intervention_fns = {}
        for percentage in range(0, 101, 10):
            probability = percentage / 100.0
            intervention_fns[f"{percentage}% Sighting-gallery/Image-query Correction"] = (
                SEEK.get_correction_policy(f"sighting_{probability}", dataset),
                SEEK.get_correction_policy(f"image_{probability}", dataset),
            )
            intervention_fns[f"{percentage}% Image Correction"] = (
                SEEK.get_correction_policy(f"image_{probability}", dataset),
                SEEK.get_correction_policy(f"image_{probability}", dataset),
            )
        intervention_fns["100% Sighting Correction"] = (
            SEEK.get_correction_policy("sighting_1.0", dataset),
            SEEK.get_correction_policy("sighting_1.0", dataset),
        )
        intervention_fns["ORACLE"] = (SEEK.oracle_correction, SEEK.oracle_correction)
        
        pr.test(
            backbone_for_concepts=MD_for_concepts, 
            backbone=MD_finedtuned, 
            concept_head=concept_head, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            intervention_fns=intervention_fns
        )
    else:
        raise ValueError(f"Unsupported pipeline: {cfg.pipeline_name}")
    

wandb.finish()
