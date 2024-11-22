import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import timm
from seek_code import SEEK



class ConceptHead:
    def __init__(
        self, crop_ears = True,  optimizer = Adam(self.layer.parameters(), lr=0.001), loss = F.mse_loss, backbone_pretraining = "savannah_elephants"
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
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        
        self.backcone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0) # Mega-Descriptor pretrained
        if backbone_pretraining == "savannah_elephants":
            state_dict = torch.load("../weights/savanna_elephants_md_v2_epoch_60.pt", map_location='cuda') # Pretrained on savannah elephants
        elif backbone_pretraining == "forest_elephants":
            state_dict = torch.load("../weights/forest_elephants-reid_weights.pt", map_location='cuda') # Pretrained on forest elephants
        self.backcone.load_state_dict(state_dict["model"]) if "optimizer" in state_dict else self.backbone.load_state_dict(state_dict) 
        
        
        self.input_dim = 768 if not crop_ears else 768 * 3  # 768 for each ear

        self.layer = nn.Sequential( 
                nn.Linear(self.input_dim, 63),
                nn.Sigmoid()
            ).to(self.device)
        
        # If some weights of the concept head exist in their location, load them
        self.c_weight_path = "../weights/chead_last_weights.pt"
        if os.path.exists(self.c_weight_path): self.layer.load_state_dict(torch.load(self.c_weight_path, map_location=self.device))

        self.optimizer = optimizer
        self.loss_fn = loss
   

    def accuracy_fn(predicted, labels):
        # Convert predicted logits to closest valid one-hot representation
        predicted_one_hot = SEEK.closest_valid_one_hot(predicted)
        
        # Initialize accuracy tracking for all attributes, including individual and averaged
        accuracies = {name: 0 for name in SEEK.attribute_names}

        index = 0
        for name in SEEK.attribute_names:
            length = SEEK.lengths[name]
            pred_slice = predicted_one_hot[:, index:index + length]
            label_slice = labels[:, index:index + length]

            # Calculate per-sample accuracy
            pred_indices = torch.argmax(pred_slice, dim=1)
            label_indices = torch.argmax(label_slice, dim=1)
            accuracies[name] = (pred_indices == label_indices).float().mean().item()

            index += length

        # Include accuracy for whole SEEK code prediction
        correct_whole_code = torch.all(predicted_one_hot == labels, dim=1).float().mean().item()
        accuracies['average'] = sum(accuracies.values()) / len(accuracies)
        accuracies['whole_code'] = correct_whole_code

        return accuracies


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
            # Access the batch, load it on the GPU
            images, _, _, _, subject_SEEK, _, left_ears, right_ears = batch[:8]
            
            labels = torch.stack([SEEK(s).one_hot_encode() for s in subject_SEEK]).to(self.device)

            with torch.no_grad() if not training else torch.enable_grad():
                # Embed the images and ears
                embeddings = self.backbone(images.to(self.device))

                if self.crop_ears:
                    # Embed the ears
                    left_embeddings = self.backbone(left_ears.to(self.device))
                    right_embeddings = self.backbone(right_ears.to(self.device))

                    # Concatenate the embeddings
                    embeddings = torch.cat((embeddings, left_embeddings, right_embeddings), dim=1)

                # Forward pass
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
        # Reinitialize the wieghts and biases of self.layer
        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_accuracy = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.epoch_pass(train_loader, training=True)
            val_loss, val_acc = self.epoch_pass(val_loader, training=False)

            # Save best weights based on validation accuracy
            if val_acc["average"] > best_val_accuracy:
                best_val_accuracy = val_acc["average"]
                best_weights = self.layer.state_dict()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc['average']:.4f}, Val Acc: {val_acc['average']:.4f}")

            # Save weights at the end of each epoch
            torch.save(best_weights, filepath)
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
                torch.cat(accumulate_outputs, dim=0))_
