import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModel
from retrieval import Retrieval
import torch.nn as nn
from pytorch_metric_learning import losses
import torch.optim as optim
import pandas as pd
import timm



class Backbone(nn.Module):
    def __init__(self, model_name="MegaDescriptor", with_ears=True, pretraining="savannah_elephants", lr=1e-4, experiment_code=None):
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"
            super().__init__()

            self.model_name = model_name
            self.with_ears = with_ears

            # Load the selected model
            if model_name == "MegaDescriptor":
                self.layer = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
            elif model_name == "MiewID-msv3":
                self.layer = AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            # Load pretraining weights if provided (only for MegaDescriptor)
            if model_name == "MegaDescriptor" and pretraining is not None:
                weight_path = Path(__file__).parent.parent / f"weights/{pretraining}.pt"
                state_dict = torch.load(weight_path, map_location=self.device)

                # Ensure compatibility with the loaded model
                if "model" in state_dict:
                    self.layer.load_state_dict(state_dict["model"], strict=False)
                else:
                    self.layer.load_state_dict(state_dict, strict=False)

            self.layer.to(self.device)
            self.freeze()

            # Optimizer & Loss Function
            self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)
            self.loss_fn = losses.ArcFaceLoss(num_classes=310, embedding_size=2304, margin=0.5, scale=64)
            self.loss_optimizer = optim.Adam(self.loss_fn.parameters(), lr=1e-5)

            # Create experiment directory
            self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_dir = Path(__file__).parent.parent / 'experiments' / f'exp_{self.exp_code}'
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

            with open(self.experiment_dir / 'note.txt', 'w') as f:
                f.write(f'---Backbone: {model_name}---\n')


    def freeze(self):
        """Freezes the model's parameters to prevent unnecessary training."""
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        self.layer.train()
        # Unfreeze the backbone parameters
        for param in self.layer.parameters():
            param.requires_grad = True
    

    def forward(self, images, left_ears = None, right_ears = None):

        embeddings = self.layer(images)
            
        if self.with_ears:
            left_embeddings = self.layer(left_ears.to(self.device))
            right_embeddings = self.layer(right_ears.to(self.device))
            embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

        return embeddings


    def epoch_pass(self, loader, training = True):

        # Are we training ?
        self.layer.train() if training else self.layer.eval()

        total_loss, total_accuracy, total_batches = 0, 0, 0
        r = Retrieval(self.exp_code)

        for batch_idx, batch in enumerate(loader):
            images, ele_id_label, subject_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[6].to('cuda'), batch[7].to('cuda')

            embeddings = self.forward(images, left_ears, right_ears)

            loss = self.loss_fn(embeddings, ele_id_label)
            
            mat = r.cosine_similarity_matrix(embeddings)
            total_accuracy += r.compute_recall_at_k(mat, ele_id_label, ele_id_label, 1)


            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                self.loss_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.loss_optimizer.step()
                
            # Compute metrics
            total_batches += 1
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_accuracy / total_batches

        return epoch_loss, epoch_accuracy

    def train(self, train_loader, val_loader, num_epochs=10):
        self.unfreeze()
        print('Backbone parameters require gradients:', any(param.requires_grad for param in self.layer.parameters()))

        history = []
        best_train_recall = 0
        ret = Retrieval(experiment_code='temp')

        for epoch in range(num_epochs):
            
            # train one epoch on train_loader
            train_loss, batch_recall = self.epoch_pass(train_loader, training=True)
            epoch_train_recall = ret.one_out_retrieval(self, train_loader)
            epoch_val_recall = ret.evaluate_model(model=self, train_loader=train_loader, test_loader=val_loader)

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]))
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "BATCH-Recall@1": batch_recall*100, **{f"Recall@{k}": v*100 for k, v in epoch_train_recall.items()}})            

            # Save best model weights
            if epoch_train_recall[1] > best_train_recall:
                best_train_recall = epoch_train_recall[1]
                best_weights = self.layer.state_dict()

        # Save best weights
        torch.save(best_weights, self.experiment_dir / 'backbone_w.pt')

        # Convert history to a DataFrame and save as CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / 'backbone_training_history.csv', index=False)

    
    def collect_embeddings(self, loader,    ba = None, ch = None, backbone_for_concepts=None, backbone=None, concept_head= None, intervention_fn = None):
        
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        self.layer.eval()

        for batch in loader:
            images, ele_id_label, _, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[6].to('cuda'), batch[7].to('cuda')
            
            with torch.no_grad():
                embeddings = self.forward(images, left_ears, right_ears)   

                collected_embeddings = torch.cat((collected_embeddings, embeddings))
                collected_labels = torch.cat((collected_labels, ele_id_label))
        
        return collected_embeddings, collected_labels