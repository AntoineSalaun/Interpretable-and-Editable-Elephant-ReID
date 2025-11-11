from seek_code import SEEK
from backbone import Backbone
from retrieval import Retrieval
from projector import Projector
from data_handler import EleHandler
from concept_head_tunneled import ConceptHeadTunneled, categorical_CE_loss, CrossedHeadNN, MultiHeadNN

from pathlib import Path
from torch.utils.data import DataLoader, Subset
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, required=True, help='Experiment type')
parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
parser.add_argument('--network_size', type=str, default='small', help='Size of the projector network: small or large')
parser.add_argument('--dataset', type=str, default='mara', help='Dataset to use: zooniverse or mara')
args = parser.parse_args()

if args.dataset == 'zooniverse': #Needs to be fixed
    dataset = EleHandler(subset='IDI_6')
    train_indices, test_indices = dataset.split_perpendicular_to_elephants_and_encounters(split_sizes=[0.5,0.5], hour_delta = 0.2)

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=True)
elif args.dataset == 'mara':
    dataset = EleHandler(subset='2+encounters', dictonary_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/mara/image_dictionary_optimized.csv', dataset_type='mara')
    train_indices, test_indices = dataset.split_parallel_to_encounters()

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=True) 

# python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py  --epochs 200 --experiment train_concept_head_on_full_mara
if args.experiment == 'train_concept_head_on_full_mara': # Train the concepthead on the full dataset
    code = 'training_concept_head_on_full_mara'

    full_mara_dataset = EleHandler(subset=None, dictonary_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/mara/image_dictionary_optimized.csv', dataset_type='mara')
    train_indices, test_indices = full_mara_dataset.split_parallel_to_encounters()
    train_subset, test_subset = Subset(full_mara_dataset, train_indices), Subset(full_mara_dataset, test_indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=True)

    backbone_for_concepts = Backbone(model_name="MegaDescriptor", pretraining=None, experiment_code=args.experiment, num_classes=len(full_mara_dataset.ele_id_to_label))
    backbone_for_concepts.unfreeze()

    CH = ConceptHeadTunneled(lr=5e-6, loss=categorical_CE_loss, experiment_code = args.experiment, layer = CrossedHeadNN(), reset_weights=True, pretraining=None)
    CH.train(train_loader, test_loader, backbone=backbone_for_concepts, num_epochs=args.epochs, predict_ele_SEEK = False)
    CH.test(test_loader, backbone=backbone_for_concepts, predict_ele_SEEK = False)

elif args.experiment == 'baseline_1': # Testing MegaDescriptor out of the box
    code = args.experiment
    MD_out_of_the_box = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code, lr=5e-6)
    MD_out_of_the_box.test(train_loader, test_loader)

elif args.experiment == 'baseline_2_1': # Finetuning MegaDescriptor on the whole dataset

    #TODO
    x=0

elif args.experiment == 'baseline_2_2':  # python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py --code MARA2+_MDfinetuning --epochs 200 --experiment mara2+_MD_finetuning

    MD_finetuned = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=args.code, lr=5e-6, num_classes=531)
    MD_finetuned.train(train_loader, test_loader, num_epochs=args.epochs)
    MD_finetuned.test(train_loader, test_loader)

elif args.experiment == 'baseline_2_3': # Finetuning MegaDescriptor through the projector architecture

    code = '[SEP-EXP]baseline_2_3_for_' + args.epochs.__str__() +'epochs_(sanity_check)'

    MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
    MD_savannah = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code)
    pr = Projector(loss_type ='ArcFace', lr=5e-6, scale=64, margin=0.5, experiment_code=code, intervention_fn=None, alpha = 0)
    concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code = code, layer = CrossedHeadNN(), reset_weights=False)

    # Freeze other models and unfreeze the projector
    MD_for_concepts.freeze()
    MD_savannah.unfreeze()
    concept_head.freeze()
    pr.unfreeze()

    # Train the model
    pr.train(
        train_loader, 
        test_loader, 
        backbone_for_concepts=MD_for_concepts, 
        backbone=MD_savannah, 
        concept_head=concept_head, 
        num_epochs=args.epochs)

    test_intervention_fns = {
    "0% Correction": None,
    "0% Correction Hard": SEEK.hard
    }   

    # Run the structured test method
    pr.test(
        backbone_for_concepts=MD_for_concepts, 
        backbone=MD_savannah, 
        concept_head=concept_head, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        intervention_fns=test_intervention_fns
    )

elif args.experiment == 'exp_1_1': # First training of CHAIR, no correction at training time

    code = '[800epochs]exp_1.1' #python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py --epochs 800 --experiment exp_1_1

    MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
    MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code)
    pr = Projector(loss_type ='ArcFace', lr=5e-6, scale=64, margin=0.5, experiment_code=code, intervention_fn=None, alpha = 0.5)
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
        num_epochs=args.epochs)

    # Define intervention functions for testing
    test_intervention_fns = {
        "ORACLE": SEEK.oracle_correction,
        "100% Correction": SEEK.perfect_correction,
        "50% Correction + hard": SEEK.correct_or_hard,
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
    
elif args.experiment == 'exp_2_1': #python /data/vision/beery/scratch/antoine/CBM_reid/methods/exp_baselines.py --epochs 800 --experiment exp_2_1
    code = '[800epochs]exp_2.1'

    MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
    MD_finedtuned = Backbone(model_name="MegaDescriptor", pretraining="savannah_elephants", experiment_code=code)
    pr = Projector(loss_type ='ArcFace', lr=5e-6, scale=64, margin=0.5, experiment_code=code, intervention_fn=SEEK.perfect_correction, alpha = 0.5, network_type=args.network_size)
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
        num_epochs=args.epochs)

    # Define intervention functions for testing
    test_intervention_fns = {
        "ORACLE": SEEK.oracle_correction,
        "100% Correction": SEEK.perfect_correction,
        "50% Correction + hard": SEEK.correct_or_hard,
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


else:
    raise ValueError('Experiment not recognized')



