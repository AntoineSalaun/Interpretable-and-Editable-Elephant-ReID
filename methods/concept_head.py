import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import timm
from seek_code import SEEK
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import wandb


class ConceptHead:
    def __init__(self, crop_ears=True, lr=0.001, loss=F.mse_loss, print_every=1, experiment_code = None, architecture = None, reset_weights = True):
        """
        Initializes the ConceptHead model.

        Args:
        - lr (float): Learning rate for the optimizer.
        - loss (callable): Loss function used during training.
        - print_every (int): Number of epochs between printing training status.
        - expermient_code (string): Where to save the logs of the experiment

        Attributes:
        - layer (nn.Module): A fully connected layer for SEEK code prediction.
        - optimizer (torch.optim.Optimizer): Optimizer for training.
        - loss_fn (callable): Loss function.
        - c_weight_path (str): Path for saving/loading concept head weights.
        """
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.crop_ears = crop_ears
        self.print_every = print_every

        self.input_dim = 768 if not crop_ears else 768 * 3  # Handle concatenated embeddings
        if architecture is None:
            self.layer = nn.Sequential(
                nn.Linear(self.input_dim, 63), 
                nn.Sigmoid()
                ).to(self.device)
        else: self.layer = architecture.to(self.device)

        self.layer.load_state_dict(torch.load(Path(__file__).parent.parent / "weights/concept_w.pt", map_location=self.device, weights_only=False)) if (Path(__file__).parent.parent / "weights/concept_w.pt").exists() and reset_weights == False else None

        
        self.loss_fn = loss
        self.lr = lr

        # Create experiment directory
        exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = Path(__file__).parent.parent / 'experiments' / f'exp_{exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)


        # start a new wandb run to track this script
        wandb.init(
            mode="online",
            # set the wandb project where this run will be logged
            project="improve-concept-head",

            # track hyperparameters and run metadata
            config={
                        "learning_rate": lr,
                        "network":self.layer,
                        "code": experiment_code,
                        "loss" : loss,
                        "optimizer": self.optimizer
                    },
            dir = self.experiment_dir
        )


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
        accuracies['left_ear'] = (accuracies['L_tear_1'] + accuracies['L_hole_1'] + accuracies['L_tear_2'] + accuracies['L_hole_2'] + accuracies['left_extreme'])/5
        accuracies['right_ear'] = (accuracies['R_tear_1'] + accuracies['R_hole_1'] + accuracies['R_tear_2'] + accuracies['R_hole_2'] + accuracies['right_extreme'])/5

        return accuracies

    def epoch_pass(self, loader, backbone, training=True, predict_ele_SEEK = False):
        """
        Processes one epoch of training or validation.

        Args:
        - loader (DataLoader): DataLoader for training or validation data.
        - training (bool): If True, trains the model. Otherwise, evaluates it.

        Returns:
        - average_loss (float): Average loss over the epoch.
        - epoch_accuracies (dict): Dictionary of accuracies for each attribute and overall metrics.
        """
        backbone.layer.eval()
        self.layer.train() if training else self.layer.eval()
        total_loss = 0
        num_batches = len(loader)
        epoch_accuracies = {name: 0 for name in SEEK.attribute_names}
        epoch_accuracies.update({"average": 0, "whole_code": 0, "left_ear": 0, "right_ear": 0})

        for batch in loader:
            
            images, _, _, _, subject_SEEK_1h, ele_SEEK_1h, left_ear, right_ear, _, _, _ = batch
            
            if predict_ele_SEEK:
                labels = ele_SEEK_1h.to(self.device)
            else:
                labels = subject_SEEK_1h.to(self.device)            
            images = images.to(self.device)

            embeddings = backbone.forward(images, left_ear, right_ear)

            outputs = self.layer(embeddings)
            #print('output shape', outputs.shape, 'labels shape', labels.shape)
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


    def train(self, train_loader, val_loader, backbone, num_epochs=10, predict_ele_SEEK = False):
        """
        Trains the ConceptHead model.

        Args:
        - train_loader (DataLoader): DataLoader for training data.
        - val_loader (DataLoader): DataLoader for validation data.
        - num_epochs (int): Number of training epochs.

        Saves:
        - A CSV file containing training and validation metrics.
        """

        print(f"Training ConceptHead for {num_epochs} epochs, reseting weights, and the backbone parameters are,", any(param.requires_grad for param in backbone.layer.parameters()), ' concept head parameters are ', any(param.requires_grad for param in self.layer.parameters()))
        self.optimizer = Adam(list(backbone.layer.parameters()) + list(self.layer.parameters()), self.lr)


        # Initialize weights for Linear layers
        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

        # List to store training history
        history = []
        best_val_accuracy = 0

        for epoch in range(num_epochs):
            # Perform a training and validation pass
            train_loss, train_acc = self.epoch_pass(train_loader, backbone, training=True, predict_ele_SEEK = predict_ele_SEEK)
            val_loss, val_acc = self.epoch_pass(val_loader, backbone, training=False, predict_ele_SEEK = predict_ele_SEEK)

            # Save best model weights
            if val_acc["average"] > best_val_accuracy:
                best_val_accuracy = val_acc["average"]
                best_weights = self.layer.state_dict()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc_avg": train_acc["average"],
                "val_acc_avg": val_acc["average"],
            })

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc_avg": train_acc["average"], "val_acc_avg": val_acc["average"], "train_acc_whole_code": train_acc["whole_code"], "val_acc_whole_code": val_acc["whole_code"]})

            # Optionally print progress
            if epoch % self.print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - Train Acc: {train_acc['average'] * 100:.2f}%, Val Acc: {val_acc['average'] * 100:.2f}% - Train whole-code: {train_acc['whole_code'] * 100:.2f}%, Val whole-code: {val_acc['whole_code'] * 100:.2f}% - Train left ear: {train_acc['left_ear'] * 100:.2f}%, Val left ear: {val_acc['left_ear'] * 100:.2f}% - Train right ear: {train_acc['right_ear'] * 100:.2f}%, Val right ear: {val_acc['right_ear'] * 100:.2f}%")

        # Save best weights
        torch.save(best_weights, self.experiment_dir / 'concept_w.pt')
        wandb.log_artifact(self.experiment_dir / 'concept_w.pt', name="concept_w.pt", type="model")

        self.layer.load_state_dict(best_weights)

        # Convert history to a DataFrame and save as CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / 'concept_training_history.csv', index=False)

        return history_df

    def test(self, test_loader, backbone, predict_ele_SEEK = False):
        """
        Tests the ConceptHead model on a test set.

        Args:
        - test_loader (DataLoader): DataLoader for testing data.

        Returns:
        - test_loss (float): Loss on the test set.
        - test_acc (dict): Accuracy metrics on the test set.
        """
        test_loss, test_acc = self.epoch_pass(test_loader, backbone, training=False, predict_ele_SEEK = predict_ele_SEEK)
        
        wandb.summary["test_loss"] = test_loss
        wandb.summary["test_acc"] = test_acc["average"]
        wandb.summary["test_acc_whole_code"] = test_acc["whole_code"]
        wandb.summary["all_test_acc"] = test_acc

        print(f"ConceptHead - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['average'] * 100:.2f}% - Test whole-code: {test_acc['whole_code'] * 100:.2f}%")

        print(f"ConceptHead - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['average'] * 100:.2f}%")
        
        with open(self.experiment_dir / 'concept_head_test_accuracy.txt', 'w') as f:
            for name, value in test_acc.items():
                accuracy_str = f"{name} Accuracy: {value * 100:.2f}%"
                print(accuracy_str)
                f.write(accuracy_str + '\n')
        
        torch.save(self.layer.state_dict(), self.experiment_dir / 'concept_w.pt')

        wandb.finish()

        return test_loss, test_acc

    def infer_all(self, dataset, backbone):
        """
        Performs inference on the entire dataset.

        Args:
        - dataset: Dataset for inference.

        Returns:
        - Tuple containing images, embeddings, predictions, true labels, and raw outputs.
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        accumulate_images, accumulate_embeddings, accumulate_predictions, accumulate_labels, accumulate_outputs = [], [], [], [], []

        for batch in dataloader:
            images, subject_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[4], batch[6].to(self.device), batch[7].to(self.device)
            
            embeddings = backbone.forward(images, left_ears, right_ears)

            with torch.no_grad():
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

    def freeze(self):
        """
        Freezes the ConceptHead parameters.
        """
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the ConceptHead parameters.
        """
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True

    def collect_embeddings(self, loader, backbone, intervention_fn = None):
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        self.layer.eval()
        print('collecting concepts from concept head')
        for batch in tqdm(loader):
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[2].to(self.device), batch[4], batch[5], batch[6].to(self.device), batch[7].to(self.device)

            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)

                outputs = self.layer(embeddings)

                predicted_SEEK = SEEK.closest_valid_one_hot(outputs)

                if intervention_fn is not None:
                    predicted_SEEK = intervention_fn(predicted_SEEK, subject_SEEK, ele_SEEK)
                
                predicted_SEEK = predicted_SEEK.to(self.device)

                subject_SEEK = torch.stack([SEEK(s).one_hot_encode() for s in subject_SEEK]).to(self.device)

                collected_embeddings = torch.cat((collected_embeddings, predicted_SEEK))
                collected_labels = torch.cat((collected_labels, ele_id_label))

        return collected_embeddings, collected_labels
 






















if __name__ == "__main__":
    ch = ConceptHead()
