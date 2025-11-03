import pandas as pd
from datetime import datetime
import pathlib
import torch
from torch.utils.data import DataLoader
from retrieval import Retrieval
import torch.nn as nn
import torch.optim as optim
from seek_code import SEEK
from pytorch_metric_learning import losses, miners
from retrieval import Retrieval
from pathlib import Path
import pandas as pd
import logging, sys


class Projector(nn.Module):
    def __init__(self, loss_type = 'ArcFace', lr=5e-5, margin=0.5, scale=64, experiment_code=None, reset_weights=True, intervention_fn=None, alpha = 0.5, print_every = 5, network_type='small'):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if network_type =='small':
            self.layer = nn.Sequential(
            nn.Linear(63, 2304),
            nn.LeakyReLU()
            ).to('cuda')
        elif network_type =='medium':
            self.layer = nn.Sequential(
                nn.Linear(63, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 2304),
                nn.LeakyReLU()
            ).to(self.device)
        else: raise('No correct network type provided')

        self.lr = lr
        self.intervention_fn = intervention_fn
        self.print_every = print_every

        if reset_weights is False:
            if (Path(__file__).parent.parent / "weights/projector_w.pt").exists():
                state_dict = torch.load(Path(__file__).parent.parent / "weights/projector_w.pt", map_location=self.device, weights_only=False)
                self.layer.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)
            else:
                raise FileNotFoundError('No projector weights found')
        
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)

        self.loss_type = loss_type
        if loss_type == 'ArcFace':
            self.loss_fn = losses.ArcFaceLoss(num_classes=310, embedding_size=2304, margin=margin, scale=scale)
            self.loss_optimizer = torch.optim.Adam(self.loss_fn.parameters(), lr=1e-4)
        elif loss_type == 'TripletLoss':
            self.miner = miners.BatchHardMiner()
            self.loss_fn = losses.TripletMarginLoss(margin=margin)
        else: raise('Incorrect Loss !!')


        self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = pathlib.Path.cwd().parent / 'experiments' / f'exp_{self.exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha

        self.logger = self.make_logger(self.experiment_dir / 'projector_log.txt')
        self.logger.info(f'-------- Initializing Projector: {network_type}---')
        self.logger.info(f'Learning Rate: {lr}')
        self.logger.info(f'Print Every: {print_every}')
        self.logger.info(f'Loss Type: {loss_type}')
        if loss_type == 'ArcFace':
            self.logger.info(f'Margin: {margin}, Scale: {scale}')
        elif loss_type == 'TripletLoss':
            self.logger.info(f'Margin: {margin}')
        self.logger.info(f'Weights reset: {reset_weights}')
        self.logger.info(f'Intervention function used during training: {self.intervention_fn.__name__ if self.intervention_fn else "None"}')
        self.logger.info(f'Alpha (weight of the projector in the final embeddings): {self.alpha}')
        if reset_weights is False:
            self.logger.info(f'Loading weights from: {Path(__file__).parent.parent / "weights/projector_w.pt"}')
        self.logger.info(f'Saving results to: {self.experiment_dir}\n')


    def compute_recall_at_k(self, embeddings, labels, k=1):
        r = Retrieval(self.exp_code)
        mat = r.similarity_matrix(embeddings)
        return r.compute_recall_at_k(mat, labels, labels, k)
        
    
    def epoch_pass(self, loader, backbone_for_concepts, backbone, ch, training=True, intervention_fn = None):

        self.layer.train() if training else self.layer.eval()
        backbone.layer.train() if training else backbone.layer.eval()
        backbone_for_concepts.layer.eval()
        ch.layer.eval()
    
        total_loss, total_accuracy, total_batches = 0, 0, 0
        
        for batch_idx, batch in enumerate(loader):               
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')
            
            #with torch.no_grad():
            embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears) 
            if self.alpha !=1: 
                embeddings = backbone.forward(images, left_ears, right_ears)
            # Compute the concepts from the backbone embeddings
            concept_logits = ch.layer(embeddings_for_concepts)
            
            #hard_predicted_concepts = SEEK.closest_valid_one_hot(concept_logits)
            
            # Allow for intervention
            if intervention_fn is not None:
                #print('We apply an intervention function')
                edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK)
            else: 
                #print('No intervention function applied, edited_concepts = hard_predicted_concepts')
                edited_concepts = concept_logits

            # Projecting the concepts back to the embeddings space
            projected_concepts = self.layer(edited_concepts.to('cuda'))

            #if batch_idx == 1 and self.alpha!=1: print(f"Embeddings norm: {torch.norm(embeddings, dim=1).mean()}, Projected concepts norm: {torch.norm(projected_concepts, dim=1).mean()}")

            if self.alpha!=1:
                edited_embeddings = (1-self.alpha) * embeddings + self.alpha * projected_concepts 
            else:
                edited_embeddings = projected_concepts

            if self.loss_type == 'ArcFace':
                loss = self.loss_fn(edited_embeddings, ele_id_label)
            elif self.loss_type == 'TripletLoss':
                # Use miner to generate triplets
                hard_triplets = self.miner(edited_embeddings, ele_id_label)
                loss = self.loss_fn(edited_embeddings, ele_id_label, hard_triplets)

            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                if self.loss_type == 'ArcFace': self.loss_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.loss_type == 'ArcFace': self.loss_optimizer.step()
                
            # Compute metrics
            total_accuracy += self.compute_recall_at_k(edited_embeddings, ele_id_label, k=1)
            total_batches += 1
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_accuracy / total_batches

        return epoch_loss, epoch_accuracy

    def collect_embeddings(self, loader,  backbone, backbone_for_concepts,  concept_head, intervention_fn = None, aggregate_seeks=False, aggregate_rules = None):
        collected_edited_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')
        
        self.layer.eval()
        backbone.layer.eval()
        backbone_for_concepts.layer.eval()
        concept_head.layer.eval()
        
        if aggregate_seeks is False:
            for batch in loader:
                images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')

                with torch.no_grad():
                    embeddings = backbone.forward(images, left_ears, right_ears)   
                    embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears)  
                    
                    concept_logits = concept_head.layer(embeddings_for_concepts)
                    
                    edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK) if intervention_fn is not None else concept_logits
                    # editied concepts are always hard 

                    projected_concepts = self.layer(edited_concepts.to('cuda'))
                    
                    if self.alpha !=1:
                        edited_embeddings = (1-self.alpha) * embeddings + self.alpha * projected_concepts 
                    else:
                        edited_embeddings = projected_concepts

                collected_edited_embeddings = torch.cat((collected_edited_embeddings, edited_embeddings))
                collected_labels = torch.cat((collected_labels, ele_id_label))
        else:
            collected_concepts = torch.tensor([]).to('cuda')
            collected_embeddings = torch.tensor([]).to('cuda')

            # First, collect all edited concepts 
            for batch in loader:
                images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')

                with torch.no_grad():
                    embeddings = backbone.forward(images, left_ears, right_ears)   
                    embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears)  
                    
                    concept_logits = concept_head.layer(embeddings_for_concepts)
                    
                    edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK) if intervention_fn is not None else concept_logits
                    # editied concepts are always hard 

                collected_embeddings = torch.cat((collected_embeddings, embeddings))
                collected_labels = torch.cat((collected_labels, ele_id_label))
                collected_concepts = torch.cat((collected_concepts, edited_concepts.to('cuda')))
            
            # Now, aggregate the concepts according to the provided rules
            subject_seek, aggregated_concepts, collected_labels = SEEK.aggregate_seek(collected_concepts, collected_labels, rules = aggregate_rules)
            
            # Create a DataLoader for the aggregated concepts
            aggregated_concepts_loader = DataLoader(list(zip(collected_embeddings, aggregated_concepts, collected_labels)), batch_size=loader.batch_size, shuffle=False)
            
            # Now, project the aggregated concepts back to the embeddings space
            for batch in aggregated_concepts_loader:
                embeddings, aggregated_concepts, ele_id_label = batch[0].to('cuda'),  batch[1].to('cuda'), batch[2].to('cuda')
                
                with torch.no_grad():
                    projected_concepts = self.layer(aggregated_concepts.to('cuda'))

                    if self.alpha !=1:
                        edited_embeddings = (1-self.alpha) * embeddings + self.alpha * projected_concepts 
                    else:
                        edited_embeddings = projected_concepts
                    collected_edited_embeddings = torch.cat((collected_edited_embeddings, edited_embeddings))

        return collected_edited_embeddings, collected_labels
    
        
    def train(self, train_loader, val_loader, backbone_for_concepts, backbone, concept_head, num_epochs=10):
        self.logger.info(f'============ STARTING TRAINING for {num_epochs} epochs - Projector parameters require gradients: {any(param.requires_grad for param in self.layer.parameters())} with intervention function: {self.intervention_fn.__name__ if self.intervention_fn else "None"} ')
        self.logger.info(f'Parameters requiring gradients - Projector: {any(p.requires_grad for p in self.layer.parameters())}, Backbone_for_concepts: {any(p.requires_grad for p in backbone_for_concepts.layer.parameters())}, Backbone: {any(p.requires_grad for p in backbone.layer.parameters())}, Concept_head: {any(p.requires_grad for p in concept_head.layer.parameters())}')
        best_val_recall = 0
        retrieval = Retrieval(experiment_code=self.exp_code)

        trainable_params = list(p for p in self.layer.parameters() if p.requires_grad) + list(p for p in backbone.layer.parameters() if p.requires_grad)
        self.optimizer = optim.Adam(trainable_params, lr=self.lr)

        for epoch in range(num_epochs):
            train_loss, batch_recall = self.epoch_pass(train_loader, backbone_for_concepts, backbone, concept_head, training=True, intervention_fn=self.intervention_fn)
            epoch_train_recall = retrieval.one_out_retrieval(model=self, loader=train_loader, backbone_for_concepts=backbone_for_concepts, backbone=backbone,  ch=concept_head)
            
            epoch_val_recall = retrieval.evaluate_model(model=self, train_loader=train_loader, test_loader=val_loader, model_to_evaluate= 'projector', ba=backbone, ch=concept_head, backbone_for_concepts=backbone_for_concepts, print_results=False, intervention_fn=self.intervention_fn)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]))
               
            if epoch_val_recall[1] > best_val_recall:
                self.logger.info('new best weights')
                best_val_recall = epoch_val_recall[1]
                best_weights = self.layer.state_dict()
                best_backbone_weights = backbone.layer.state_dict()
        
        # Save best weights
        torch.save(best_weights, self.experiment_dir / 'projector_w.pt')
        self.layer.load_state_dict(best_weights)
        torch.save(best_backbone_weights, self.experiment_dir / 'backbone_w.pt')
        backbone.layer.load_state_dict(best_backbone_weights)
        
        self.logger.info('============ TRAINING COMPLETE ============')
        self.logger.info(f'Best Val Recall@1: {best_val_recall*100:.2f}%')
        self.logger.info(f'Weights saved to: {self.experiment_dir / "projector_w.pt"} and {self.experiment_dir / "backbone_w.pt"}')
        

    def make_logger(self, log_path: Path, level=logging.INFO, overwrite=False):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger_name = f"projector.{log_path}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            if overwrite:
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()
            else:
                return logger

        mode = 'w' if overwrite else 'a'
        file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def test(self, backbone_for_concepts, backbone, concept_head, train_loader, test_loader, intervention_fns):
        """
        Test the model with different intervention functions and store results as summaries.
        """

        self.logger.info(f'Testing with intervention functions: {", ".join(intervention_fns.keys())}')
        results_log = self.make_logger(self.experiment_dir / 'projector_test_results.txt', overwrite=True)

        results = {}
        retrieval = Retrieval(experiment_code=self.exp_code)

        for aggregate_gallery_seeks in [True,False]:
            for name, fn in intervention_fns.items():
                self.logger.info(f'Retrieval with {name} correction and aggregate_gallery_seeks={aggregate_gallery_seeks}-------------------------------------')
                metrics = retrieval.evaluate_model(
                    model=self,
                    train_loader=train_loader,
                    test_loader=test_loader, model_to_evaluate= 'projector',
                    ba=backbone,
                    ch=concept_head,
                    backbone_for_concepts=backbone_for_concepts,
                    intervention_fn=fn,
                    aggregate_gallery_seeks=aggregate_gallery_seeks,
                    show_matches=False, 
                    show_plot=False, 
                    show_tsne=False, 
                    print_results=False
                )
                results[name] = metrics
                
                self.logger.info(f'Results with {name} correction: ' + ", ".join([f"Recall@{k}: {v * 100:.2f}%" for k, v in metrics.items()]))
                results_log.info("\n".join([f"Recall@{k}: {v * 100:.2f}%" for k, v in metrics.items()]) + "\n")
    
        return results



    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True
        

