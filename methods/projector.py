import pandas as pd
from datetime import datetime
import pathlib
import torch
from torch.utils.data import DataLoader
from retrieval import Retrieval
import torch.nn as nn
import torch.optim as optim
from seek_code import SEEK
from pytorch_metric_learning import losses
from retrieval import Retrieval
from pathlib import Path
import wandb



class Projector(nn.Module):
    def __init__(self, lr=5e-5, margin=0.5, scale=64, experiment_code=None, reset_weights=True, intervention_fn=None):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layer = nn.Sequential(
            nn.Linear(63, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2304),
            nn.LeakyReLU()
        ).to(self.device)

        self.lr = lr
        self.intervention_fn = intervention_fn

        if reset_weights is False:
            if (Path(__file__).parent.parent / "weights/projector_w.pt").exists():
                state_dict = torch.load(Path(__file__).parent.parent / "weights/projector_w.pt", map_location=self.device, weights_only=False)
                self.layer.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)
            else:
                raise FileNotFoundError('No projector weights found')
        
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)
        self.loss_fn = losses.ArcFaceLoss(num_classes=310, embedding_size=2304, margin=margin, scale=scale)
        self.loss_optimizer = torch.optim.Adam(self.loss_fn.parameters(), lr=1e-4)

        self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = pathlib.Path.cwd().parent / 'experiments' / f'exp_{self.exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb with intervention function used in training
        wandb.init(
            project="projector_training",
            name=self.exp_code,
            config={
                "learning_rate": self.lr,
                "margin": self.loss_fn.margin,
                "scale": self.loss_fn.scale,
                "training_intervention_fn": self.intervention_fn.__name__ if self.intervention_fn else "None"
            }
        )
        wandb.define_metric("val-Recall@1", summary="max")


    def compute_recall_at_k(self, embeddings, labels, k=1):
        r = Retrieval(self.exp_code)
        mat = r.cosine_similarity_matrix(embeddings)
        return r.compute_recall_at_k(mat, labels, labels, k)
        
    
    def epoch_pass(self, loader, backbone_for_concepts, backbone, ch, training=True, intervention_fn = None):

        total_loss, total_accuracy, total_batches = 0, 0, 0
        
        for batch_idx, batch in enumerate(loader):
                    # Are we training ?
            self.layer.train() if training else self.layer.eval()   
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')
            
            #with torch.no_grad():
            embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears)  
            embeddings = backbone.forward(images, left_ears, right_ears)
            # Compute the concepts from the backbone embeddings
            concept_logits = ch.layer(embeddings_for_concepts)
            
            hard_predicted_concepts = SEEK.closest_valid_one_hot(concept_logits)
            
            # Allow for intervention
            if intervention_fn is not None:
                #print('We apply an intervention function')
                edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK)
            else: 
                #print('No intervention function applied, edited_concepts = hard_predicted_concepts')
                edited_concepts = hard_predicted_concepts

            # Projecting the concepts back to the embeddings space
            projected_concepts = self.layer(edited_concepts.to('cuda'))

            # We add the concepts projected back to the intiial embeddings space
            edited_embeddings = projected_concepts + embeddings

            #print('ele_id_label:', ele_id_label)

            loss = self.loss_fn(edited_embeddings, ele_id_label)

            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                self.loss_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_optimizer.step()
                
            # Compute metrics
            total_accuracy += self.compute_recall_at_k(edited_embeddings, ele_id_label, k=1)
            total_batches += 1
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_accuracy / total_batches

        return epoch_loss, epoch_accuracy

    def collect_embeddings(self, loader,  backbone, backbone_for_concepts,  concept_head, intervention_fn = None):
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')
        
        self.layer.eval()
        backbone.layer.eval()
        concept_head.layer.eval()
        
        for batch in loader:
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')

            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)   
                embeddings_for_concepts = backbone_for_concepts.forward(images, left_ears, right_ears)  
                concept_logits = concept_head.layer(embeddings_for_concepts)
                #hard_concepts = SEEK.closest_valid_one_hot(concept_logits)
                edited_concepts = intervention_fn(concept_logits, subject_SEEK, ele_SEEK) if intervention_fn is not None else concept_logits
                edited_concepts = edited_concepts.to('cuda')
                projected_concepts = self.layer(edited_concepts)
                edited_embeddings = projected_concepts + embeddings

            collected_embeddings = torch.cat((collected_embeddings, edited_embeddings))
            collected_labels = torch.cat((collected_labels, ele_id_label))

        return collected_embeddings, collected_labels
    
        
    def train(self, train_loader, val_loader, backbone_for_concepts, backbone, concept_head, num_epochs=10):
        history = []
        best_train_recall = 0
        retrieval = Retrieval(experiment_code=self.exp_code)
        print(f'Parameters requiring gradients - Projector: {any(p.requires_grad for p in self.layer.parameters())}, Backbone_for_concepts: {any(p.requires_grad for p in backbone_for_concepts.layer.parameters())}, Backbone: {any(p.requires_grad for p in backbone.layer.parameters())}, Concept_head: {any(p.requires_grad for p in concept_head.layer.parameters())}')
        self.optimizer = optim.Adam(list(self.layer.parameters())+list(backbone.layer.parameters()), lr=self.lr)

        for epoch in range(num_epochs):
            train_loss, batch_recall = self.epoch_pass(train_loader, backbone_for_concepts, backbone, concept_head, training=True)
            epoch_train_recall = retrieval.one_out_retrieval(model=self, backbone_for_concepts=backbone_for_concepts, backbone=backbone, loader=train_loader, ch=concept_head)

            epoch_val_recall = retrieval.evaluate_model(model=self, train_loader=train_loader, test_loader=val_loader, ba=backbone, ch=concept_head, backbone_for_concepts=backbone_for_concepts, print_results=False  )
            #backbone_val_recall = retrieval.evaluate_model(model=backbone, train_loader=train_loader, test_loader=val_loader  )
            
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "BATCH-Recall@1": batch_recall * 100, **{f"train-Recall@{k}": v * 100 for k, v in epoch_train_recall.items()}, **{f"val-Recall@{k}": v * 100 for k, v in epoch_val_recall.items()}})
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "BATCH-Recall@1": batch_recall * 100, **{f"train-Recall@{k}": v * 100 for k, v in epoch_train_recall.items()}, **{f"val-Recall@{k}": v * 100 for k, v in epoch_val_recall.items()}})
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]))
            
            if epoch_train_recall[1] > best_train_recall:
                best_train_recall = epoch_train_recall[1]
                best_weights = self.layer.state_dict()
        
        torch.save(best_weights, self.experiment_dir / 'projector_w.pt')
        pd.DataFrame(history).to_csv(self.experiment_dir / 'projector_training_history.csv', index=False)
        wandb.summary.update({f"Final-Train-Recall@{k}": v * 100 for k, v in epoch_train_recall.items()})

    def test(self, backbone_for_concepts, backbone, concept_head, train_loader, test_loader, intervention_fns):
        """
        Test the model with different intervention functions and store results as summaries.
        """
        results = {}
        retrieval = Retrieval(experiment_code=self.exp_code)

        for name, fn in intervention_fns.items():
            print(f'Retrieval with {name} correction-------------------------------------')
            metrics = retrieval.evaluate_model(
                model=self,
                train_loader=train_loader,
                test_loader=test_loader,
                ba=backbone,
                ch=concept_head,
                backbone_for_concepts=backbone_for_concepts,
                intervention_fn=fn,
                show_matches=False, show_plot=False, show_tsne=False, print_results=False
            )
            
            results[name] = metrics
            
            print(f"{name} - " + ", ".join([f"Recall@{k}: {v * 100:.2f}%" for k, v in metrics.items()]))
        
        # Store results as summaries within the existing wandb run
        wandb.summary.update({
            f"{name}-Recall@{k}": v * 100 for name, metrics in results.items() for k, v in metrics.items()
        })
        
        wandb.finish()
        return results



    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True
        

