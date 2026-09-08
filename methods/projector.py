import copy
import pandas as pd
from datetime import datetime
import pathlib, wandb
import torch
from retrieval import Retrieval
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses, miners
from retrieval import Retrieval
from pathlib import Path
import pandas as pd
import logging, sys


class Projector(nn.Module):
    def __init__(self, loss_type = 'ArcFace', lr=5e-5, margin=0.5, scale=64, experiment_code=None, reset_weights=True, intervention_fn=None, alpha = 0.5, print_every = 5, network_type='small', num_classes=310, wd = 1e-5, layer_norm=False, alpha_learnable=False):
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
        self.wd = wd
        self.intervention_fn = intervention_fn
        self.print_every = print_every

        if reset_weights is False:
            if (Path(__file__).parent.parent / "weights/projector_w.pt").exists():
                state_dict = torch.load(Path(__file__).parent.parent / "weights/projector_w.pt", map_location=self.device, weights_only=False)
                self.layer.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)
            else:
                raise FileNotFoundError('No projector weights found')
        
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr, weight_decay=wd)

        self.loss_type = loss_type
        if loss_type == 'ArcFace':
            self.loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=2304, margin=margin, scale=scale)
            self.loss_optimizer = torch.optim.Adam(self.loss_fn.parameters(), lr=1e-4)
        elif loss_type == 'TripletLoss':
            self.miner = miners.BatchHardMiner()
            self.loss_fn = losses.TripletMarginLoss(margin=margin)
        else: raise('Incorrect Loss !!')


        self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = pathlib.Path.cwd().parent / 'experiments' / f'exp_{self.exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_learnable = alpha_learnable
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, device=self.device))
        else:
            self.alpha = alpha
        self.layer_norm = layer_norm

        wandb.summary.update({"status": "Projector initialized",
                              "Projector/learning_rate": lr,
                              "Projector/loss_type": loss_type,
                              "Projector/Weights reset": reset_weights,
                              "Projector/intervention_fn_at_training": self.intervention_fn.__name__ if self.intervention_fn else "None",
                              "Projector/alpha": alpha,
                              "Projector/Loading weights from": str(Path(__file__).parent.parent / "weights/projector_w.pt") if reset_weights is False else "N/A",
                              "Projector/saving_results_to": str(self.experiment_dir),
                              "Projector/network_type": network_type,
                            "Projector/weight_decay": wd
                              })


    def epoch_pass(self, loader, backbone_for_concepts, backbone, ch, training=True, intervention_fn=None):
        if training:
            self.layer.train()
            backbone.layer.train()
        else:
            self.layer.eval()
            backbone.layer.eval()
        backbone_for_concepts.layer.eval()
        ch.layer.eval()
        if self.alpha_learnable:
            self.alpha.requires_grad = training

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            images = batch[0].to(self.device)
            ele_id_label = batch[2].to(self.device)
            subject_SEEK = batch[4]
            ele_SEEK = batch[5]
            left_ears = batch[6].to(self.device)
            right_ears = batch[7].to(self.device)
            idx = batch[10]

            # Frozen models don't need gradients
            with torch.no_grad():
                embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears)
                concept_logits = ch.layer(embeddings_for_concepts)

            embeddings = backbone.forward(images, left_ears, right_ears)
            if self.layer_norm:
                embeddings = torch.nn.functional.layer_norm(embeddings, embeddings.size()[1:])
            
            #hard_predicted_concepts = SEEK.closest_valid_one_hot(concept_logits)
            
            # Allow for intervention
            if intervention_fn is not None:
                edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK, ele_id=ele_id_label, idx=idx)
            else: 
                edited_concepts = concept_logits

            # Projecting the concepts back to the embeddings space
            projected_concepts = self.layer(edited_concepts.to('cuda'))
            if self.layer_norm == True: 
                projected_concepts = torch.nn.functional.layer_norm(projected_concepts, projected_concepts.size()[1:])

            edited_embeddings = (1-self.alpha) * embeddings + self.alpha * projected_concepts

            if self.loss_type == 'ArcFace':
                loss = self.loss_fn(edited_embeddings, ele_id_label)
            elif self.loss_type == 'TripletLoss':
                # Use miner to generate triplets
                hard_triplets = self.miner(edited_embeddings, ele_id_label)
                loss = self.loss_fn(edited_embeddings, ele_id_label, hard_triplets)

            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                if self.loss_type == 'ArcFace':
                    self.loss_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.loss_type == 'ArcFace':
                    self.loss_optimizer.step()

            # Fast in-batch recall computation (no Retrieval object needed)
            with torch.no_grad():
                embeddings_norm = torch.nn.functional.normalize(edited_embeddings, dim=1)
                sim_matrix = embeddings_norm @ embeddings_norm.T
                sim_matrix.fill_diagonal_(-float('inf'))
                top1_indices = sim_matrix.argmax(dim=1)
                correct = (ele_id_label[top1_indices] == ele_id_label).sum().item()
                total_correct += correct
                total_samples += images.size(0)

            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return epoch_loss, epoch_accuracy

    def collect_embeddings(self, loader,  backbone, backbone_for_concepts,  concept_head, intervention_fn = None):
        collected_edited_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')
        
        self.layer.eval()
        backbone.layer.eval()
        backbone_for_concepts.layer.eval()
        concept_head.layer.eval()
        
        for batch in loader:
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears, idx = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda'), batch[10]

            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)
                embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears)
                concept_logits = concept_head.layer(embeddings_for_concepts)
                edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK, ele_id=ele_id_label, idx=idx) if intervention_fn is not None else concept_logits
                projected_concepts = self.layer(edited_concepts.to('cuda'))

                if self.layer_norm:
                    embeddings = torch.nn.functional.layer_norm(embeddings, embeddings.size()[1:])
                    projected_concepts = torch.nn.functional.layer_norm(projected_concepts, projected_concepts.size()[1:])

                edited_embeddings = (1-self.alpha) * embeddings + self.alpha * projected_concepts

            collected_edited_embeddings = torch.cat((collected_edited_embeddings, edited_embeddings))
            collected_labels = torch.cat((collected_labels, ele_id_label))

        return collected_edited_embeddings, collected_labels
    
    def save_weights(self, backbone, epoch, recall_at_1):
        wandb.summary.update({"saved_weights": f"Saving backbone and projector weights at epoch {epoch} with recall@1={recall_at_1}, in {self.experiment_dir}"})
        weights = copy.deepcopy(self.layer.state_dict())
        backbone_weights = copy.deepcopy(backbone.layer.state_dict())
        torch.save(weights, self.experiment_dir / f'projector.pt')
        torch.save(backbone_weights, self.experiment_dir / f'backbone_w.pt')
        if self.alpha_learnable:
            alpha_val = self.alpha.item()
            torch.save({'alpha': alpha_val, 'epoch': epoch, 'val_recall_at_1': recall_at_1}, self.experiment_dir / f'alpha.pt')
            wandb.save(str(self.experiment_dir / f'alpha.pt'), base_path=str(self.experiment_dir))
        wandb.save(str(self.experiment_dir / f'projector.pt'), base_path=str(self.experiment_dir))
        wandb.save(str(self.experiment_dir / f'backbone_w.pt'), base_path=str(self.experiment_dir))
        return weights, backbone_weights


    def train(self, train_loader, val_loader, backbone_for_concepts, backbone, concept_head, num_epochs=10, save_best=True, save_embeddings=False, min_best_epoch=10):
        wandb.summary.update({"status": f"Projector training started for {num_epochs} epochs",
                              "Pr/projector_parameters_require_grad": any(param.requires_grad for param in self.layer.parameters()),
                              "Pr/intervention_fn_at_training": self.intervention_fn.__name__ if self.intervention_fn else "None",
                              "Pr/backbone_for_concepts_parameters_require_grad": any(p.requires_grad for p in backbone_for_concepts.layer.parameters()),
                              "Pr/backbone_parameters_require_grad": any(p.requires_grad for p in backbone.layer.parameters()),
                              "Pr/concept_head_parameters_require_grad": any(p.requires_grad for p in concept_head.layer.parameters())})
        
        best_val_recall = 0
        best_epoch = None
        best_alpha = None
        weights = None
        backbone_weights = None
        saved_checkpoint = False
        retrieval = Retrieval(experiment_code=self.exp_code)

        trainable_params = list(p for p in self.layer.parameters() if p.requires_grad) + list(p for p in backbone.layer.parameters() if p.requires_grad)
        if self.alpha_learnable:
            trainable_params.append(self.alpha)
        self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.wd)
        
        # Debug: Log optimizer configuration
        num_projector_params = sum(p.numel() for p in self.layer.parameters() if p.requires_grad)
        num_backbone_params = sum(p.numel() for p in backbone.layer.parameters() if p.requires_grad)

        wandb.summary.update({
            "Pr/optimizer_lr": self.lr,
            "Pr/optimizer_wd": self.wd,
            "Pr/num_projector_params": num_projector_params,
            "Pr/num_backbone_params": num_backbone_params,
            "Pr/num_trainable_param_tensors": len(trainable_params),
            "Pr/save_best": save_best,
            "Pr/min_best_epoch": min_best_epoch,
        })



        for epoch in range(num_epochs):
            current_epoch = epoch + 1

            train_loss, batch_recall = self.epoch_pass(train_loader, backbone_for_concepts, backbone, concept_head, training=True, intervention_fn=self.intervention_fn)

            # Build log dict with basic metrics
            log_dict = {
                "Pr/Epoch": current_epoch,
                "Pr/Train Loss": train_loss,
                "Pr/Batch Recall@1": batch_recall * 100,
            }

            if current_epoch % self.print_every == 0 or current_epoch == num_epochs:
                # Only run expensive retrieval evaluations every print_every epochs
                epoch_train_recall = retrieval.one_out_retrieval(model=self, loader=train_loader, backbone_for_concepts=backbone_for_concepts, backbone=backbone, ch=concept_head)
                epoch_val_recall = retrieval.evaluate_model(model=self, train_loader=train_loader, test_loader=val_loader, model_to_evaluate='projector', ba=backbone, ch=concept_head, backbone_for_concepts=backbone_for_concepts, print_results=False, intervention_fn=self.intervention_fn)
                
                print(f"Epoch {current_epoch}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]))
                with open(self.experiment_dir / 'projector_log.txt', 'a') as f:
                    f.write(f"Epoch {current_epoch}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]) + "\n")
                
                # Add recall metrics to log dict
                log_dict.update({f"Pr/Train Recall@{k}": v*100 for k,v in epoch_train_recall.items()})
                log_dict.update({f"Pr/Val Recall@{k}": v*100 for k,v in epoch_val_recall.items()})
                if self.alpha_learnable:
                    log_dict["Pr/Alpha"] = self.alpha.item()

                can_save_best = current_epoch >= min_best_epoch
                if save_best and can_save_best:
                    if epoch_val_recall[1] > best_val_recall:
                        if self.alpha_learnable:
                            best_alpha = self.alpha.item()
                        best_epoch = current_epoch
                        best_val_recall = epoch_val_recall[1]
                        weights, backbone_weights = self.save_weights(backbone, current_epoch, epoch_val_recall[1])
                        saved_checkpoint = True
                        if save_embeddings:
                            emb, labels = self.collect_embeddings(train_loader,  backbone, backbone_for_concepts,  concept_head, intervention_fn = self.intervention_fn)
                            torch.save({'embeddings': emb.cpu(), 'labels': labels.cpu()}, self.experiment_dir / f'projector_embeddings_epoch_{current_epoch}.pt')
                            wandb.save(str(self.experiment_dir / f'projector_embeddings.pt'), base_path=str(self.experiment_dir))            
                            
                elif not save_best and can_save_best:
                    best_epoch = current_epoch
                    best_val_recall = epoch_val_recall[1]
                    if self.alpha_learnable:
                        best_alpha = self.alpha.item()
                    weights, backbone_weights = self.save_weights(backbone, current_epoch, epoch_val_recall[1])
                    saved_checkpoint = True

            # Single wandb.log call per epoch with all metrics
            wandb.log(log_dict)
                    
            
        # Load best weights from saved files (more robust than in-memory references)
        if save_best and saved_checkpoint and (self.experiment_dir / 'projector.pt').exists():
            print(f"[INFO] Loading best weights from {self.experiment_dir / 'projector.pt'} (best_val_recall@1={best_val_recall:.4f})")
            self.layer.load_state_dict(torch.load(self.experiment_dir / 'projector.pt', map_location=self.device))
            backbone.layer.load_state_dict(torch.load(self.experiment_dir / 'backbone_w.pt', map_location=self.device))
            if self.alpha_learnable and (self.experiment_dir / 'alpha.pt').exists():
                alpha_data = torch.load(self.experiment_dir / 'alpha.pt', map_location=self.device)
                self.alpha.data.fill_(alpha_data['alpha'])

        summary_dict = {"status": "Projector training complete",
                        "best_val_recall_at_1": best_val_recall,
                        "Pr/Best Val Recall@1": best_val_recall * 100,
                        "Pr/Best Epoch": best_epoch,
                        "Pr/checkpoint_saved": saved_checkpoint,
                        "projector_weights_path": str(self.experiment_dir / 'projector.pt'),
                        "backbone_weights_path": str(self.experiment_dir / 'backbone_w.pt')
                        }
        if self.alpha_learnable:
            summary_dict.update({
                "Pr/Best Alpha": best_alpha,
                "Pr/Best Alpha Epoch": best_epoch,
                "learnt_alpha": best_alpha,
                "learned_alpha": best_alpha,
            })

        wandb.summary.update(summary_dict)
        
        

    def test(self, backbone_for_concepts, backbone, concept_head, train_loader, test_loader, intervention_fns):
        """
        Test the model with different intervention functions and store results as summaries.
        """

        wandb.log({"status": "Testing with intervention functions", "projector_test_intervention_functions": list(intervention_fns.keys())})

        results = {}
        retrieval = Retrieval(experiment_code=self.exp_code)
        policy_summary = {}

        for name, fn in intervention_fns.items():
            if isinstance(fn, tuple):
                gallery_fn, query_fn = fn
            else:
                gallery_fn = query_fn = fn

            policy_summary[name] = {
                "gallery": self._correction_summary(gallery_fn),
                "query": self._correction_summary(query_fn),
            }

            wandb.summary.update({"status": f"Pr/Retrieval with {name} correction"})
            metrics = retrieval.evaluate_model(
                model=self,
                train_loader=train_loader,
                test_loader=test_loader, model_to_evaluate= 'projector',
                ba=backbone,
                ch=concept_head,
                backbone_for_concepts=backbone_for_concepts,
                gallery_intervention_fn=gallery_fn,
                query_intervention_fn=query_fn,
                show_matches=False,
                show_plot=False,
                show_tsne=False,
                print_results=False
            )
            results[name] = metrics

            # Not tested (used to do wandb.log here)
            wandb.summary.update({f"Pr/Test Recall@{k} with {name} correction": v * 100 for k, v in metrics.items()})

        wandb.summary.update({"projector_test_correction_policies": policy_summary})
        return results


    @staticmethod
    def _correction_summary(fn):
        return {
            "name": getattr(fn, "__name__", str(fn)),
            "scope": getattr(fn, "correction_scope", None),
            "probability": getattr(fn, "correction_probability", None),
            "selected_units": len(getattr(fn, "corrected_units", []) or []),
        }


    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True
        
