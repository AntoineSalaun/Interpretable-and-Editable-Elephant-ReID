import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pytorch_metric_learning import losses
from transformers import AutoModel

from retrieval import Retrieval


class MiewID(nn.Module):
    def __init__(
        self,
        pretraining="out_of_box",
        lr=5e-6,
        wd=0.0,
        experiment_code=None,
        print_every=5,
        num_classes=310,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = Path.cwd() / "experiments" / f"exp_{self.exp_code}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = Path(__file__).parent.parent / "weights"
        self.print_every = print_every

        self.layer = AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)
        loaded_weights = "Using out-of-box conservationxlabs/miewid-msv3 weights"
        if pretraining is not None and str(pretraining).lower() not in {"none", "out_of_box", "zero_shot"}:
            state_dict, weight_path = self.load_pretraining(pretraining)
            self.layer.load_state_dict(state_dict, strict=False)
            loaded_weights = f"Loaded weights from {weight_path}"

        self.layer.to(self.device)
        self.freeze()
        self.embedding_size = self.infer_embedding_size()

        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr, weight_decay=wd)
        self.loss_fn = losses.ArcFaceLoss(
            num_classes=int(num_classes),
            embedding_size=self.embedding_size,
            margin=0.5,
            scale=64,
        ).to(self.device)
        self.loss_optimizer = optim.Adam(self.loss_fn.parameters(), lr=1e-5)

        wandb.summary.update(
            {
                "MiewID/pretraining": str(pretraining),
                "MiewID/loaded_weights": loaded_weights,
                "MiewID/device": self.device,
                "MiewID/embedding_size": self.embedding_size,
                "MiewID/params/total": sum(p.numel() for p in self.layer.parameters()),
                "MiewID/experiment_dir": str(self.experiment_dir),
                "MiewID/lr": lr,
                "MiewID/wd": wd,
            }
        )

    def load_pretraining(self, pretraining):
        candidate = Path(str(pretraining)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            weight_path = candidate
        elif (Path(__file__).parent.parent / candidate).exists():
            weight_path = Path(__file__).parent.parent / candidate
        elif candidate.suffix:
            weight_path = self.weights_dir / candidate.name
        else:
            weight_path = self.weights_dir / f"{candidate.name}.pt"

        if not weight_path.exists():
            raise FileNotFoundError(f"Could not find MiewID pretraining weights for '{pretraining}'")

        state_dict = torch.load(weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        return state_dict, weight_path

    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True

    @staticmethod
    def extract_embeddings(outputs):
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, dict):
            for key in ("pooler_output", "image_embeds", "embeddings", "last_hidden_state"):
                if key in outputs:
                    value = outputs[key]
                    return value[:, 0] if value.dim() == 3 else value
        for key in ("pooler_output", "image_embeds", "embeddings", "last_hidden_state"):
            value = getattr(outputs, key, None)
            if value is not None:
                return value[:, 0] if value.dim() == 3 else value
        if isinstance(outputs, (tuple, list)) and outputs:
            value = outputs[0]
            return value[:, 0] if value.dim() == 3 else value
        raise TypeError(f"Could not extract MiewID embeddings from output type {type(outputs)}")

    def forward(self, images):
        images = images.to(self.device)
        try:
            outputs = self.layer(images)
        except TypeError:
            outputs = self.layer(pixel_values=images)
        return self.extract_embeddings(outputs)

    def infer_embedding_size(self):
        hidden = getattr(getattr(self.layer, "config", None), "hidden_size", None)
        if hidden is not None:
            return int(hidden)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            return int(self.forward(dummy).shape[1])

    def collect_embeddings(self, loader, *args, **kwargs):
        was_training = self.layer.training
        self.layer.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                ele_id_label = batch[2].to(self.device)
                embeddings.append(self.forward(images).detach().cpu())
                labels.append(ele_id_label.detach().cpu())
        self.layer.train(was_training)
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).long()

    def epoch_pass(self, loader, training=True):
        self.layer.train(training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch in loader:
                images = batch[0].to(self.device)
                ele_id_label = batch[2].to(self.device)
                embeddings = self.forward(images)
                loss = self.loss_fn(embeddings, ele_id_label)

                with torch.no_grad():
                    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)
                    sim_matrix = embeddings_norm @ embeddings_norm.T
                    sim_matrix.fill_diagonal_(-float("inf"))
                    top1_indices = sim_matrix.argmax(dim=1)
                    total_correct += (ele_id_label[top1_indices] == ele_id_label).sum().item()
                    total_samples += images.size(0)

                if training:
                    self.optimizer.zero_grad()
                    self.loss_optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.loss_optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader), total_correct / total_samples if total_samples else 0.0

    def save_weights(self, epoch, recall_at_1):
        experiment_path = self.experiment_dir / "MiewID_w.pt"
        wandb.summary.update(
            {
                "MiewID/saved_weights": (
                    f"Saving MiewID weights at epoch {epoch} with recall@1={recall_at_1}, "
                    f"in {self.experiment_dir}"
                )
            }
        )
        weights = copy.deepcopy(self.layer.state_dict())
        torch.save(weights, experiment_path)
        wandb.save(str(experiment_path), base_path=str(self.experiment_dir))
        return weights

    def train(self, train_loader, val_loader, num_epochs=10, save_best=True):
        best_val_recall = -1
        best_weights = copy.deepcopy(self.layer.state_dict())
        retrieval = Retrieval(experiment_code="temp")

        initial_train_recall = retrieval.one_out_retrieval(self, train_loader)
        initial_val_recall = retrieval.evaluate_model(
            model=self,
            model_to_evaluate="backbone",
            train_loader=train_loader,
            test_loader=val_loader,
            print_results=False,
        )
        wandb.log(
            {
                "MiewID/Epoch": 0,
                **{f"MiewID/Train Recall@{k}": v * 100 for k, v in initial_train_recall.items()},
                **{f"MiewID/Val Recall@{k}": v * 100 for k, v in initial_val_recall.items()},
            }
        )
        if save_best:
            best_val_recall = initial_val_recall[1]
            best_weights = self.save_weights(0, initial_val_recall[1])

        self.unfreeze()
        for epoch in range(num_epochs):
            train_loss, batch_recall = self.epoch_pass(train_loader, training=True)
            wandb.log(
                {
                    "MiewID/Epoch": epoch + 1,
                    "MiewID/Train Loss": train_loss,
                    "MiewID/Batch Recall@1": batch_recall * 100,
                }
            )

            if epoch % self.print_every == 0:
                epoch_train_recall = retrieval.one_out_retrieval(self, train_loader)
                epoch_val_recall = retrieval.evaluate_model(
                    model=self,
                    model_to_evaluate="backbone",
                    train_loader=train_loader,
                    test_loader=val_loader,
                    print_results=False,
                )
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | Loss: {train_loss:.4f} | "
                    f"Batch R@1: {batch_recall * 100:.2f}% | Val R@1: {epoch_val_recall[1] * 100:.2f}%"
                )
                wandb.log(
                    {
                        **{f"MiewID/Train Recall@{k}": v * 100 for k, v in epoch_train_recall.items()},
                        **{f"MiewID/Val Recall@{k}": v * 100 for k, v in epoch_val_recall.items()},
                    }
                )

                if save_best:
                    if epoch_val_recall[1] > best_val_recall:
                        best_val_recall = epoch_val_recall[1]
                        best_weights = self.save_weights(epoch + 1, epoch_val_recall[1])
                else:
                    best_val_recall = max(best_val_recall, epoch_val_recall[1])
                    best_weights = self.save_weights(epoch + 1, epoch_val_recall[1])

        self.layer.load_state_dict(best_weights)
        wandb.summary.update({"MiewID/Best Val Recall@1": best_val_recall * 100})

    def test(self, train_loader, test_loader):
        retrieval = Retrieval(experiment_code=self.exp_code)
        recalls = retrieval.evaluate_model(
            model=self,
            model_to_evaluate="backbone",
            train_loader=train_loader,
            test_loader=test_loader,
        )
        wandb.summary.update(
            {
                **{f"MiewID/Test Recall@{k}": v * 100 for k, v in recalls.items()},
                **{f"Test-Recall@{k}": v * 100 for k, v in recalls.items()},
            }
        )
        return recalls
