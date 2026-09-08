import copy
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
import wandb



class Backbone(nn.Module):
    def __init__(self, model_name="MegaDescriptor", with_ears=True, pretraining="savannah_elephants", lr=5e-6, wd=0.0, experiment_code=None, print_every=5, num_classes=310, layer_norm=False):
            super().__init__()
            self.device = 'cuda' if torch.cuda.is_available() else "cpu"

            self.model_name = model_name
            self.with_ears = with_ears
            self.layer_norm = layer_norm
            self.print_every = print_every
            self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_dir = Path.cwd() / 'experiments' / f'exp_{self.exp_code}'
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self.weights_dir = Path(__file__).parent.parent / "weights"

            if model_name == "MegaDescriptor":
                build_model = lambda: timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
            elif model_name == "MiewID-msv3":
                build_model = lambda: AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            loaded_weights = 'No pretraining specified, skipping weight load'
            encoder_mode = 'shared_encoder'

            if pretraining is not None and str(pretraining).lower() != 'none' and ',' in str(pretraining):
                if not self.with_ears:
                    raise ValueError("Comma-separated pretraining requires with_ears=True")
                pretraining_list = [p.strip() for p in str(pretraining).split(',') if p.strip()]
                if len(pretraining_list) != 3:
                    raise ValueError("Pretraining must be one checkpoint or three checkpoints: complete,left,right")

                self.layer = nn.ModuleDict({
                    "whole": build_model(),
                    "left": build_model(),
                    "right": build_model(),
                })
                loaded_paths = []
                for part, weight_name in zip(["whole", "left", "right"], pretraining_list):
                    state_dict, weight_path = self.load_pretraining(weight_name)
                    self.layer[part].load_state_dict(state_dict, strict=False)
                    loaded_paths.append(f"{part}<-{weight_path}")
                loaded_weights = "Loaded separate weights: " + ", ".join(loaded_paths)
                encoder_mode = 'separate_encoders'

            else:
                self.layer = build_model()
                if pretraining is not None and str(pretraining).lower() != 'none':
                    state_dict, weight_path = self.load_pretraining(pretraining)
                    if self.with_ears and all(any(key.startswith(f"{part}.") for key in state_dict.keys()) for part in ["whole", "left", "right"]):
                        self.layer = nn.ModuleDict({
                            "whole": build_model(),
                            "left": build_model(),
                            "right": build_model(),
                        })
                        self.layer.load_state_dict(state_dict, strict=False)
                        loaded_weights = f'Loaded multi-view backbone weights from {weight_path}'
                        encoder_mode = 'separate_encoders'
                    else:
                        self.layer.load_state_dict(state_dict, strict=False)
                        loaded_weights = f'Loaded weights from {weight_path}'

            self.layer.to(self.device)
            self.freeze()

            reference_layer = self.layer["whole"] if isinstance(self.layer, nn.ModuleDict) else self.layer
            self.embedding_size = getattr(reference_layer, "num_features", 768)
            total_embedding_size = self.embedding_size * (3 if self.with_ears else 1)

            # Optimizer & Loss Function
            self.optimizer = optim.Adam(self.layer.parameters(), lr=lr, weight_decay=wd)
            self.loss_fn = losses.ArcFaceLoss(
                num_classes=int(num_classes),
                embedding_size=total_embedding_size,
                margin=0.5,
                scale=64,
            ).to(self.device)
            self.loss_optimizer = optim.Adam(self.loss_fn.parameters(), lr=1e-5)

            wandb.log({'status': f'Backbone initialized: {model_name} with pretraining: {pretraining}'})
            wandb.summary.update({
                'Backbone/loaded_weights': loaded_weights,
                'Backbone/encoder_mode': encoder_mode,
                "Backbone/optimizer": str(self.optimizer),
                "Backbone/loss_optimizer": str(self.loss_optimizer),
                "Backbone/loss_fn": str(self.loss_fn),
                "Backbone/device": self.device,
                "Backbone/params/total": sum(p.numel() for p in self.layer.parameters()),
                "Backbone/exp_code": self.exp_code,
                "Backbone/experiment_dir": str(self.experiment_dir),
                "Backbone/layer_norm": self.layer_norm,
                "Backbone/lr": lr,
                "Backbone/wd": wd,
            })


    def load_pretraining(self, pretraining):
        candidate = Path(str(pretraining)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            weight_path = candidate
        else:
            repo_candidate = Path(__file__).parent.parent / candidate
            if repo_candidate.exists():
                weight_path = repo_candidate
            elif candidate.suffix:
                weight_path = self.weights_dir / candidate.name
            else:
                weight_path = self.weights_dir / f"{candidate.name}.pt"

        if not weight_path.exists():
            raise FileNotFoundError(f"Could not find backbone pretraining weights for '{pretraining}'")

        state_dict = torch.load(weight_path, map_location='cpu')
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        return state_dict, weight_path


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

        if isinstance(self.layer, nn.ModuleDict):
            embeddings = self.layer["whole"](images.to(self.device))
        else:
            embeddings = self.layer(images.to(self.device))
            
        if self.with_ears:
            if isinstance(self.layer, nn.ModuleDict):
                left_embeddings = self.layer["left"](left_ears.to(self.device))
                right_embeddings = self.layer["right"](right_ears.to(self.device))
            else:
                left_embeddings = self.layer(left_ears.to(self.device))
                right_embeddings = self.layer(right_ears.to(self.device))
            embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

        if self.layer_norm:
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        return embeddings


    def epoch_pass(self, loader, training=True):
        # Are we training?
        if training:
            self.layer.train()
        else:
            self.layer.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Use no_grad for validation to save memory and speed up
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch in loader:
                images = batch[0].to(self.device)
                ele_id_label = batch[2].to(self.device)
                left_ears = batch[6].to(self.device)
                right_ears = batch[7].to(self.device)

                embeddings = self.forward(images, left_ears, right_ears)
                loss = self.loss_fn(embeddings, ele_id_label)

                # Fast in-batch recall approximation (no separate Retrieval object needed)
                with torch.no_grad():
                    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)
                    sim_matrix = embeddings_norm @ embeddings_norm.T
                    sim_matrix.fill_diagonal_(-float('inf'))  # Exclude self
                    top1_indices = sim_matrix.argmax(dim=1)
                    correct = (ele_id_label[top1_indices] == ele_id_label).sum().item()
                    total_correct += correct
                    total_samples += images.size(0)

                # Backward pass and optimization
                if training:
                    self.optimizer.zero_grad()
                    self.loss_optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.loss_optimizer.step()

                total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return epoch_loss, epoch_accuracy


    def save_weights(self, epoch, recall_at_1):
        experiment_path = self.experiment_dir / 'backbone_w.pt'
        wandb.summary.update({"saved_weights": f"Saving backbone weights at epoch {epoch} with recall@1={recall_at_1}, in {self.experiment_dir}"})
        weights = copy.deepcopy(self.layer.state_dict())
        torch.save(weights, experiment_path)
        wandb.save(str(experiment_path), base_path=str(self.experiment_dir))
        return weights
    
    def train(self, train_loader, val_loader, num_epochs=10, save_best = False):
        self.unfreeze()
        wandb.summary.update({'status': f'Ba/===== STRATING TRAINING BACKBONE for {num_epochs} epochs - Backbone parameters require gradients: {any(param.requires_grad for param in self.layer.parameters())}'})
        
        best_val_recall = -1
        weights = copy.deepcopy(self.layer.state_dict())
        ret = Retrieval(experiment_code='temp')

        for epoch in range(num_epochs):
            
            # train one epoch on train_loader
            train_loss, batch_recall = self.epoch_pass(train_loader, training=True)
            
            # Keep all epoch metrics in one history row so custom x-axes work in W&B.
            log_dict = {
                "Ba/Epoch": epoch + 1,
                "Ba/Train Loss": train_loss,
                "Ba/Batch Recall@1": batch_recall*100,
            }
            
            if epoch % self.print_every == 0 :
                epoch_train_recall = ret.one_out_retrieval(self, train_loader)
                epoch_val_recall = ret.evaluate_model(model=self, model_to_evaluate= 'backbone', train_loader=train_loader, test_loader=val_loader, print_results = False)
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]))
                with open(self.experiment_dir / 'backbone_log.txt', 'a') as f: f.write(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Batch R@1: {batch_recall*100:.2f}% | Train: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_train_recall.items()]) + " | Val: " + " ".join([f"R@{k}={v*100:.1f}%" for k,v in epoch_val_recall.items()]) + "\n")
                
                log_dict.update({f"Ba/Train Recall@{k}": v*100 for k,v in epoch_train_recall.items()})
                log_dict.update({f"Ba/Val Recall@{k}": v*100 for k,v in epoch_val_recall.items()})

                # Save best model weights
                if save_best:
                    if epoch_val_recall[1] > best_val_recall:
                        best_val_recall = epoch_val_recall[1]
                        weights = self.save_weights(epoch+1, epoch_val_recall[1])
                else:
                    best_val_recall = max(best_val_recall, epoch_val_recall[1])
                    weights = self.save_weights(epoch+1, epoch_val_recall[1])

            wandb.log(log_dict)

        # Save best weights
        wandb.summary.update({'Training Finished': f'Best Val Recall@1: {best_val_recall*100:.2f}%. Best weights loaded and saved to backbone_w.pt in {str(self.experiment_dir / 'backbone_w.pt')}'})
        self.layer.load_state_dict(weights)
        


    def test(self, train_loader, test_loader):
        wandb.summary.update({'status': '=== Testing Backbone  ==='})
        r = Retrieval(experiment_code = self.exp_code)
        recalls = r.evaluate_model(model = self, model_to_evaluate= 'backbone', train_loader=train_loader, test_loader=test_loader)
        
        wandb.summary.update({f'Ba/Test Recall@{k}': v*100 for k,v in recalls.items()})
        

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
