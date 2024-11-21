import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

"""
I asked ChatGPT to make a class for my concept head from the code that I wrote in the probing notebook. This class has not been debugged yet.
"""

class ConceptHead:
    def __init__(
        self, input_dim, output_dim, use_sigmoid=False, lr=1e-3, device="cuda", weights_path="best_weights.pth"
    ):
        """
        Initializes the ConceptHead class.

        Args:
        - input_dim (int): Dimension of input embeddings.
        - output_dim (int): Dimension of output predictions.
        - use_sigmoid (bool): Whether to apply sigmoid activation in the output layer.
        - lr (float): Learning rate for the optimizer.
        - device (str): Device to run the model ("cuda" or "cpu").
        - weights_path (str): Path to load/save the best model weights.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        # Define the layer
        if use_sigmoid:
            self.layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Sigmoid()
            ).to(self.device)
        else:
            self.layer = nn.Linear(input_dim, output_dim).to(self.device)

        self.optimizer = Adam(self.layer.parameters(), lr=lr)
        self.loss_fn = F.mse_loss
        self.accuracy_fn = calculate_attribute_accuracy  # Function from the notebook

        # Load pre-trained weights if they exist
        if os.path.exists(weights_path):
            self.load_weights(weights_path)

    def save_weights(self, filepath):
        """Saves the model weights to a file."""
        torch.save(self.layer.state_dict(), filepath)

    def load_weights(self, filepath):
        """Loads model weights from a file."""
        self.layer.load_state_dict(torch.load(filepath, map_location=self.device))
        self.layer.to(self.device)

    def reinitialize_weights(self):
        """Reinitializes the weights of the model."""
        for layer in self.layer.children() if isinstance(self.layer, nn.Sequential) else [self.layer]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def epoch_pass(self, loader, training=True, with_ears=False):
        """
        Processes a single epoch.

        Args:
        - loader (DataLoader): DataLoader for the current dataset.
        - training (bool): If True, trains the model. Otherwise, evaluates it.
        - with_ears (bool): Additional functionality for ear-based features.

        Returns:
        - average_loss (float): Average loss over the epoch.
        - epoch_accuracies (dict): Accuracies for all attributes and overall.
        """
        self.layer.train() if training else self.layer.eval()

        total_loss = 0
        num_batches = len(loader)
        epoch_accuracies = {name: 0 for name in SEEK.attribute_names}
        epoch_accuracies["average"] = 0
        epoch_accuracies["whole_code"] = 0

        for batch in loader:
            images, _, _, _, subject_SEEK, _, left_ears, right_ears = batch[:8]
            images = images.to(self.device)
            labels = torch.stack(
                [SEEK(s).one_hot_encode() for s in subject_SEEK]
            ).to(self.device)

            with torch.no_grad() if not training else torch.enable_grad():
                # Embed the images and ears
                embeddings = model(images)
                left_embeddings = model(left_ears.to(self.device))
                right_embeddings = model(right_ears.to(self.device))

                # Concatenate the embeddings
                total_embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

                # Forward pass
                outputs = self.layer(total_embeddings)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                predicted_SEEK = closest_valid_one_hot(outputs)
                batch_accuracies = self.accuracy_fn(predicted_SEEK, labels)
                for name, value in batch_accuracies.items():
                    epoch_accuracies[name] += value

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        average_loss = total_loss / num_batches
        epoch_accuracies = {
            name: accuracy / num_batches for name, accuracy in epoch_accuracies.items()
        }
        return average_loss, epoch_accuracies

    def train(self, train_loader, val_loader, num_epochs=10):
        """
        Trains the concept head.

        Args:
        - train_loader (DataLoader): DataLoader for training data.
        - val_loader (DataLoader): DataLoader for validation data.
        - num_epochs (int): Number of epochs.

        Returns:
        - history (dict): Training and validation losses and accuracies over epochs.
        """
        # Reinitialize weights
        self.reinitialize_weights()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_accuracy = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.epoch_pass(train_loader, training=True)
            val_loss, val_acc = self.epoch_pass(val_loader, training=False)

            # Save best weights based on validation accuracy
            if val_acc["average"] > best_val_accuracy:
                best_val_accuracy = val_acc["average"]
                self.save_weights(self.weights_path)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Train Acc: {train_acc['average']:.4f}, "
                    f"Val Acc: {val_acc['average']:.4f}"
                )

        return history

    def test(self, test_loader):
        """
        Tests the concept head.

        Args:
        - test_loader (DataLoader): DataLoader for testing data.

        Returns:
        - test_loss (float): Loss on the test set.
        - test_acc (dict): Accuracies on the test set.
        """
        test_loss, test_acc = self.epoch_pass(test_loader, training=False)
        print(f"Test Loss: {test_loss:.4f}")
        for name, value in test_acc.items():
            print(f"{name} Accuracy: {value * 100:.2f}%")
        return test_loss, test_acc

    def infer_all(self, dataset):
        """
        Infers outputs for all data in a dataset.

        Args:
        - dataset: Dataset to infer on.

        Returns:
        - accumulate_images (torch.Tensor): All images processed.
        - accumulate_embeddings (torch.Tensor): All concatenated embeddings (images + ears).
        - accumulate_predictions (torch.Tensor): Predicted SEEK one-hot codes.
        - accumulate_labels (torch.Tensor): True SEEK one-hot codes.
        - accumulate_outputs (torch.Tensor): Raw model outputs before conversion.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
        accumulate_images, accumulate_embeddings, accumulate_predictions, accumulate_labels, accumulate_outputs = [], [], [], [], []

        for batch in dataloader:
            images, subject_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[4], batch[6].to(self.device), batch[7].to(self.device)
            with torch.no_grad():
                total_embeddings = torch.cat((model(images), model(left_ears), model(right_ears)), dim=1)
                outputs = self.layer(total_embeddings)
                predicted_SEEK = closest_valid_one_hot(outputs)
                subject_SEEK = torch.stack([SEEK(s).one_hot_encode() for s in subject_SEEK]).to(self.device)

                accumulate_images.append(images.cpu())
                accumulate_embeddings.append(total_embeddings.cpu())
                accumulate_predictions.append(predicted_SEEK.cpu())
                accumulate_labels.append(subject_SEEK.cpu())
                accumulate_outputs.append(outputs.cpu())

        return (torch.cat(accumulate_images, dim=0), torch.cat(accumulate_embeddings, dim=0), 
                torch.cat(accumulate_predictions, dim=0), torch.cat(accumulate_labels, dim=0), 
                torch.cat(accumulate_outputs, dim=0))
