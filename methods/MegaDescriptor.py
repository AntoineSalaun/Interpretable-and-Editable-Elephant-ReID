import copy
from datetime import datetime
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pytorch_metric_learning import losses

from retrieval import Retrieval


class MegaDescriptor(nn.Module):
    def __init__(
        self,
        pretraining="savannah_elephants",
        lr=5e-6,
        wd=0.0,
        crop_type="complete",
        experiment_code=None,
        print_every=5,
        num_classes=310,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = Path.cwd() / "experiments" / f"exp_{self.exp_code}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.layer = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        self.embedding_size = getattr(self.layer, "num_features", 768)
        self.print_every = print_every
        self.crop_type = str(crop_type).lower()
        if self.crop_type not in {"complete", "left_ear", "right_ear"}:
            raise ValueError(
                f"Unsupported MegaDescriptor crop_type: {crop_type} "
                "(expected 'complete', 'left_ear' or 'right_ear')"
            )

        loaded_weights = "Using default MegaDescriptor weights from hf-hub"
        if pretraining is not None and str(pretraining).lower() not in {"none", "zero_shot"}:
            weight_path = Path(__file__).parent.parent / f"weights/{pretraining}.pt"
            state_dict = torch.load(weight_path, map_location=self.device)
            self.layer.load_state_dict(state_dict["model"] if "model" in state_dict else state_dict, strict=False)
            loaded_weights = f"Loaded weights from {weight_path}"

        self.layer.to(self.device)
        self.freeze()

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
                "MegaDescriptor/pretraining": str(pretraining),
                "MegaDescriptor/loaded_weights": loaded_weights,
                "MegaDescriptor/device": self.device,
                "MegaDescriptor/embedding_size": self.embedding_size,
                "MegaDescriptor/params/total": sum(p.numel() for p in self.layer.parameters()),
                "MegaDescriptor/experiment_dir": str(self.experiment_dir),
                "MegaDescriptor/lr": lr,
                "MegaDescriptor/wd": wd,
                "MegaDescriptor/crop_type": self.crop_type,
            }
        )

    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True

    def forward(self, images):
        embeddings = self.layer(images.to(self.device))
        return embeddings

    def _select_crop(self, batch):
        if self.crop_type == "complete":
            return batch[0]
        if self.crop_type == "left_ear":
            return batch[6]
        return batch[7]

    def epoch_pass(self, loader, training=True):
        if training:
            self.layer.train()
        else:
            self.layer.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images = self._select_crop(batch).to(self.device)
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

        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_accuracy

    def save_weights(self, epoch, recall_at_1):
        experiment_path = self.experiment_dir / "MegaDescriptor_w.pt"
        wandb.summary.update(
            {
                "MegaDescriptor/saved_weights": (
                    f"Saving MegaDescriptor weights at epoch {epoch} "
                    f"with recall@1={recall_at_1}, in {self.experiment_dir}"
                )
            }
        )
        weights = copy.deepcopy(self.layer.state_dict())
        torch.save(weights, experiment_path)
        wandb.save(str(experiment_path), base_path=str(self.experiment_dir))
        return weights

    def train(self, train_loader, val_loader, num_epochs=10, save_best=True):
        self.unfreeze()
        best_val_recall = -1
        best_weights = copy.deepcopy(self.layer.state_dict())
        retrieval = Retrieval(experiment_code="temp")

        wandb.summary.update(
            {
                "MegaDescriptor/status": (
                    f"Training MegaDescriptor for {num_epochs} epochs - "
                    f"parameters require gradients: {any(param.requires_grad for param in self.layer.parameters())}"
                )
            }
        )

        for epoch in range(num_epochs):
            train_loss, batch_recall = self.epoch_pass(train_loader, training=True)

            wandb.log(
                {
                    "MD/Epoch": epoch + 1,
                    "MD/Train Loss": train_loss,
                    "MD/Batch Recall@1": batch_recall * 100,
                    "Ba/Epoch": epoch + 1,
                    "Ba/Train Loss": train_loss,
                    "Ba/Batch Recall@1": batch_recall * 100,
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
                log_line = (
                    f"Epoch {epoch + 1}/{num_epochs} | Loss: {train_loss:.4f} | "
                    f"Batch R@1: {batch_recall * 100:.2f}% | Train: "
                    + " ".join([f"R@{k}={v * 100:.1f}%" for k, v in epoch_train_recall.items()])
                    + " | Val: "
                    + " ".join([f"R@{k}={v * 100:.1f}%" for k, v in epoch_val_recall.items()])
                )
                print(log_line)
                with open(self.experiment_dir / "MegaDescriptor_log.txt", "a") as f:
                    f.write(log_line + "\n")

                wandb.log(
                    {
                        **{f"MD/Train Recall@{k}": v * 100 for k, v in epoch_train_recall.items()},
                        **{f"MD/Val Recall@{k}": v * 100 for k, v in epoch_val_recall.items()},
                        **{f"Ba/Train Recall@{k}": v * 100 for k, v in epoch_train_recall.items()},
                        **{f"Ba/Val Recall@{k}": v * 100 for k, v in epoch_val_recall.items()},
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
        wandb.summary.update(
            {
                "MegaDescriptor/best_val_recall@1": best_val_recall * 100,
                "Ba/Best Val Recall@1": best_val_recall * 100,
            }
        )

    def test(self, train_loader, test_loader):
        wandb.summary.update({"MegaDescriptor/status": "Testing MegaDescriptor"})
        retrieval = Retrieval(experiment_code=self.exp_code)
        recalls = retrieval.evaluate_model(
            model=self,
            model_to_evaluate="backbone",
            train_loader=train_loader,
            test_loader=test_loader,
        )
        wandb.summary.update(
            {
                **{f"MD/Test Recall@{k}": v * 100 for k, v in recalls.items()},
                **{f"Ba/Test Recall@{k}": v * 100 for k, v in recalls.items()},
            }
        )
        return recalls

    def collect_embeddings(
        self,
        loader,
        ba=None,
        ch=None,
        backbone_for_concepts=None,
        backbone=None,
        concept_head=None,
        intervention_fn=None,
    ):
        collected_embeddings = []
        collected_labels = []

        self.layer.eval()
        for batch in loader:
            images = self._select_crop(batch).to(self.device)
            ele_id_label = batch[2].to(self.device)

            with torch.no_grad():
                embeddings = self.forward(images)
                collected_embeddings.append(embeddings)
                collected_labels.append(ele_id_label)

        return torch.cat(collected_embeddings, dim=0), torch.cat(collected_labels, dim=0)
