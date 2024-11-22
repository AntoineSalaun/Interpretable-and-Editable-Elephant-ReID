import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import timm
from seek_code import SEEK


class ConceptHead:
    def __init__(self, crop_ears=True, lr=0.001, loss=F.mse_loss, backbone_pretraining="savannah_elephants", print_every=5, save_weights=True):
        """
        Initializes the ConceptHead model.

        Args:
        - crop_ears (bool): If True, includes ear embeddings in the input.
        - lr (float): Learning rate for the optimizer.
        - loss (callable): Loss function used during training.
        - backbone_pretraining (str): Pretrained model type ('savannah_elephants' or 'forest_elephants').
        - print_every (int): Number of epochs between printing training status.
        - save_weights (bool): Whether to save model weights after training.

        Attributes:
        - backbone (nn.Module): The backbone feature extractor (e.g., Swin Transformer).
        - layer (nn.Module): A fully connected layer for SEEK code prediction.
        - optimizer (torch.optim.Optimizer): Optimizer for training.
        - loss_fn (callable): Loss function.
        - c_weight_path (str): Path for saving/loading concept head weights.
        """
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.crop_ears = crop_ears
        self.print_every = print_every
        self.save_weights = save_weights

        # Load the backbone model
        self.backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        if backbone_pretraining == "savannah_elephants":
            state_dict = torch.load("../weights/savanna_elephants_md_v2_epoch_60.pt", map_location=self.device)
        elif backbone_pretraining == "forest_elephants":
            state_dict = torch.load("../weights/forest_elephants-reid_weights.pt", map_location=self.device)
        self.backbone.load_state_dict(state_dict["model"] if "optimizer" in state_dict else state_dict)
        self.backbone.to(self.device)

        self.input_dim = 768 if not crop_ears else 768 * 3  # Handle concatenated embeddings
        self.layer = nn.Sequential(nn.Linear(self.input_dim, 63), nn.Sigmoid()).to(self.device)

        # Load weights for concept head, if available
        self.c_weight_path = "../weights/chead_last_weights.pt"
        if os.path.exists(self.c_weight_path):
            self.layer.load_state_dict(torch.load(self.c_weight_path, map_location=self.device))

        self.optimizer = Adam(self.layer.parameters(), lr)
        self.loss_fn = loss

    def accuracy_fn(self, predicted, labels):
        """
        Computes accuracy for each attribute and the whole SEEK code.

        Args:
        - predicted (torch.Tensor): Predicted logits from the model.
        - labels (torch.Tensor): Ground truth one-hot encoded labels.

        Returns:
        - accuracies (dict): Dictionary containing per-attribute accuracy, average accuracy, and whole SEEK code accuracy.
        """
        predicted_one_hot = SEEK.closest_valid_one_hot(predicted)
        accuracies = {name: 0 for name in SEEK.attribute_names}

        index = 0
        for name in SEEK.attribute_names:
            length = SEEK.lengths[name]
            pred_slice = predicted_one_hot[:, index:index + length]
            label_slice = labels[:, index:index + length]
            pred_indices = torch.argmax(pred_slice, dim=1)
            label_indices = torch.argmax(label_slice, dim=1)
            accuracies[name] = (pred_indices == label_indices).float().mean().item()
            index += length

        correct_whole_code = torch.all(predicted_one_hot == labels, dim=1).float().mean().item()
        accuracies['average'] = sum(accuracies.values()) / len(accuracies)
        accuracies['whole_code'] = correct_whole_code
        return accuracies

    def epoch_pass(self, loader, training=True):
        """
        Processes one epoch of training or validation.

        Args:
        - loader (DataLoader): DataLoader for training or validation data.
        - training (bool): If True, trains the model. Otherwise, evaluates it.

        Returns:
        - average_loss (float): Average loss over the epoch.
        - epoch_accuracies (dict): Dictionary of accuracies for each attribute and overall metrics.
        """
        self.layer.train() if training else self.layer.eval()
        total_loss = 0
        num_batches = len(loader)
        epoch_accuracies = {name: 0 for name in SEEK.attribute_names}
        epoch_accuracies["average"] = 0
        epoch_accuracies["whole_code"] = 0

        for batch in loader:
            if self.crop_ears:
                images, _, _, _, subject_SEEK_1h, _, left_ears, right_ears = batch[:8]
            else:
                images, _, _, _, subject_SEEK_1h = batch[:5]

            labels = subject_SEEK_1h.to(self.device)
            images = images.to(self.device)

            with torch.no_grad() if not training else torch.enable_grad():
                embeddings = self.backbone(images)
                if self.crop_ears:
                    left_embeddings = self.backbone(left_ears.to(self.device))
                    right_embeddings = self.backbone(right_ears.to(self.device))
                    embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

                outputs = self.layer(embeddings)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                batch_accuracies = self.accuracy_fn(outputs, labels)
                for name, value in batch_accuracies.items():
                    epoch_accuracies[name] += value

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        average_loss = total_loss / num_batches
        epoch_accuracies = {name: accuracy / num_batches for name, accuracy in epoch_accuracies.items()}
        return average_loss, epoch_accuracies

    def train(self, train_loader, val_loader, num_epochs=10):
        """
        Trains the ConceptHead model.

        Args:
        - train_loader (DataLoader): DataLoader for training data.
        - val_loader (DataLoader): DataLoader for validation data.
        - num_epochs (int): Number of training epochs.

        Returns:
        - history (dict): Training and validation metrics across all epochs.
        """
        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_accuracy = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.epoch_pass(train_loader, training=True)
            val_loss, val_acc = self.epoch_pass(val_loader, training=False)

            if val_acc["average"] > best_val_accuracy:
                best_val_accuracy = val_acc["average"]
                best_weights = self.layer.state_dict()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % self.print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - Train Acc: {train_acc['average'] * 100:.2f}%, Val Acc: {val_acc['average'] * 100:.2f}%")

        if self.save_weights:
            torch.save(best_weights, self.c_weight_path)
        return history

    def test(self, test_loader):
        """
        Tests the ConceptHead model on a test set.

        Args:
        - test_loader (DataLoader): DataLoader for testing data.

        Returns:
        - test_loss (float): Loss on the test set.
        - test_acc (dict): Accuracy metrics on the test set.
        """
        test_loss, test_acc = self.epoch_pass(test_loader, training=False)
        print(f"Test Loss: {test_loss:.4f}")
        for name, value in test_acc.items():
            print(f"{name} Accuracy: {value * 100:.2f}%")
        return test_loss, test_acc

    def infer_all(self, dataset):
        """
        Performs inference on the entire dataset.

        Args:
        - dataset: Dataset for inference.

        Returns:
        - Tuple containing images, embeddings, predictions, true labels, and raw outputs.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
        accumulate_images, accumulate_embeddings, accumulate_predictions, accumulate_labels, accumulate_outputs = [], [], [], [], []

        for batch in dataloader:
            images, subject_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[4], batch[6].to(self.device), batch[7].to(self.device)
            with torch.no_grad():
                embeddings = self.backbone(images)
                if self.crop_ears:
                    left_embeddings = self.backbone(left_ears)
                    right_embeddings = self.backbone(right_ears)
                    embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

                outputs = self.layer(embeddings)

                predicted_SEEK = SEEK.closest_valid_one_hot(outputs)
                subject_SEEK = torch.stack([SEEK(s).one_hot_encode() for s in subject_SEEK]).to(self.device)

                accumulate_images.append(images.cpu())
                accumulate_embeddings.append(embeddings.cpu())
                accumulate_predictions.append(predicted_SEEK.cpu())
                accumulate_labels.append(subject_SEEK.cpu())
                accumulate_outputs.append(outputs.cpu())

        return (torch.cat(accumulate_images, dim=0), torch.cat(accumulate_embeddings, dim=0), 
                torch.cat(accumulate_predictions, dim=0), torch.cat(accumulate_labels, dim=0), 
                torch.cat(accumulate_outputs, dim=0))
