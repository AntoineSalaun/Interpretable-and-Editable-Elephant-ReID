import pandas as pd
from datetime import datetime
import pathlib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from seek_code import SEEK
from pytorch_metric_learning import losses
from retrieval import Retrieval
from pathlib import Path
import wandb


class Projector(nn.Module):
    def __init__(self, criterion = 'ArcFace', lr = 1e-4, margin=0.5, scale=64, experiment_code = None, reset_weights = True):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layer =    nn.Sequential(
                        nn.Linear(63, 512),
                        nn.LeakyReLU(),
                        nn.Linear(512, 2304),
                        nn.LeakyReLU()
        ).to(self.device)
        
        #self.layer = nn.Sequential(
        #    nn.Linear(63, 2304).to(self.device),
            #nn.BatchNorm1d(2304).to(self.device),
        #    nn.LeakyReLU().to(self.device)
        #)

        # start a new wandb run to track this script
        wandb.init(
            mode="online",
            # set the wandb project where this run will be logged
            project="CBM-reid-Projector-debug",

            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "loss": criterion,
            "Arface-Margin": margin,
            "Arface-Scale": scale,
            }
        )

        self.lr = lr

        if reset_weights is False:
            if (Path(__file__).parent.parent / "weights/projector_w.pt").exists():
                state_dict = torch.load(Path(__file__).parent.parent / "weights/projector_w.pt", map_location=self.device, weights_only=False)
                self.layer.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)
            else: raise FileNotFoundError('No projector weights found')
            
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)

        self.criterion = criterion
        if criterion == 'cross-entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif criterion == 'ArcFace':
            self.loss_fn = losses.ArcFaceLoss(num_classes=310, embedding_size=2304, margin=margin, scale=scale)
            self.loss_optimizer = torch.optim.Adam(self.loss_fn.parameters(), lr=1e-4)

        self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = pathlib.Path.cwd().parent / 'experiments' / f'exp_{self.exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.experiment_dir / 'note.txt', 'a') as f: 
            f.write('---projector---')

    def compute_recall_at_k(self, embeddings, labels, k=1):
        r = Retrieval(self.exp_code)
        mat = r.cosine_similarity_matrix(embeddings)
        return r.compute_recall_at_k(mat, labels, labels, k)
        
    
    def epoch_pass(self, loader, ba, ch, cl = None, training=True, intervention_fn = None):

        total_loss, total_accuracy, total_batches = 0, 0, 0
        
        for batch_idx, batch in enumerate(loader):
                    # Are we training ?
            self.layer.train() if training else self.layer.eval()   
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')
            
            #with torch.no_grad():
            embeddings = ba.forward(images, left_ears, right_ears)  
            # Compute the concepts from the backbone embeddings
            concept_logits = ch.layer(embeddings)
            
            hard_predicted_concepts = SEEK.closest_valid_one_hot(concept_logits)
            #print('Hard predicted concepts:', SEEK(hard_predicted_concepts[0]))
            # Allow for intervention
            if intervention_fn is not None:
                #print('We apply an intervention function')
                edited_concepts = intervention_fn(hard_predicted_concepts, subject_SEEK, ele_SEEK)
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
                if self.criterion == 'ArcFace': self.loss_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.criterion == 'ArcFace': self.loss_optimizer.step()
                
            # Compute metrics
            total_accuracy += self.compute_recall_at_k(edited_embeddings, ele_id_label, k=1)
            total_batches += 1
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_accuracy / total_batches

        return epoch_loss, epoch_accuracy

    def collect_embeddings(self, loader, backbone, concept_head, intervention_fn = None):
        self.layer.eval()

        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')
        
        self.layer.eval()
        backbone.layer.eval()
        concept_head.layer.eval()
        
        for batch in loader:
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[5], batch[6].to('cuda'), batch[7].to('cuda')

            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)   
                concept_logits = concept_head.layer(embeddings)
                edited_concepts = SEEK.closest_valid_one_hot(concept_logits)
                edited_concepts = intervention_fn(edited_concepts, subject_SEEK, ele_SEEK) if intervention_fn is not None else edited_concepts
                edited_concepts = edited_concepts.to('cuda')
                projected_concepts = self.layer(edited_concepts)
                edited_embeddings = projected_concepts + embeddings

            collected_embeddings = torch.cat((collected_embeddings, edited_embeddings))
            collected_labels = torch.cat((collected_labels, ele_id_label))

        

        return collected_embeddings, collected_labels
        

    def train(self, train_loader, val_loader,  backbone , concept_head, classifier = None, num_epochs=10, intervention_fn = None):

        print('Concept head parameters require gradients:', any(param.requires_grad for param in concept_head.layer.parameters()),  '\nBackbone parameters require gradients:', any(param.requires_grad for param in backbone.layer.parameters()), '\nProjector parameters require gradients:', any(param.requires_grad for param in self.layer.parameters()))

        self.optimizer = torch.optim.Adam(list(backbone.layer.parameters()) + list(concept_head.layer.parameters()) + list(self.layer.parameters()), lr=self.lr)

        history = []
        best_train_recall = 0
        ret = Retrieval(experiment_code='temp')

        for epoch in range(num_epochs):
            train_loss, batch_recall = self.epoch_pass(train_loader, backbone, ch=concept_head, cl=classifier, training=True, intervention_fn = intervention_fn)
            
            epoch_train_recall = ret.one_out_retrieval(self, train_loader, ba=backbone, ch=concept_head, intervention_fn = intervention_fn)

            epoch_val_recall = ret.one_out_retrieval(self, val_loader, ba=backbone, ch=concept_head, intervention_fn = intervention_fn)   

            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "batch_recall@1": batch_recall * 100,
                **{f"epoch_train_recall@{k}": v * 100 for k, v in epoch_train_recall.items()},
                **{f"epoch_val_recall@{k}": v * 100 for k, v in epoch_val_recall.items()}
            })

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, BATCH-Recall@1: {batch_recall*100:.2f}%, --- TRAIN-EPOCH - Recall@1: {epoch_train_recall[1]*100:.2f}%, Recall@5: {epoch_train_recall[5]*100:.2f}%, Recall@20: {epoch_train_recall[20]*100:.2f}%, Recall@100: {epoch_train_recall[100]*100:.2f}% --- VAL-EPOCH - Recall@1: {epoch_val_recall[1]*100:.2f}%, Recall@5: {epoch_val_recall[5]*100:.2f}%, Recall@20: {epoch_val_recall[20]*100:.2f}%, Recall@100: {epoch_val_recall[100]*100:.2f}%")
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "BATCH-Recall@1": batch_recall*100, **{f"train-Recall@{k}": v*100 for k, v in epoch_train_recall.items()}, **{f"val-Recall@{k}": v*100 for k, v in epoch_val_recall.items()}})            

            # Save best model weights
            if epoch_train_recall[1] > best_train_recall:
                best_train_recall = epoch_train_recall[1]
                best_weights = self.layer.state_dict()

        # Save best weights
        torch.save(best_weights, self.experiment_dir / 'projector_w.pt')
        torch.load(self.experiment_dir / 'projector_w.pt')
        wandb.save(str(self.experiment_dir / 'best_projector_w.pt'))

        # Convert history to a DataFrame and save as CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / 'projector_training_history.csv', index=False)

        wandb.finish()


    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True
        

