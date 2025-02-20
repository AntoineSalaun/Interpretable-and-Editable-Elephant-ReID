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



class ConceptHeadTunneled:
    def __init__(self, crop_ears=True, lr=0.001, loss=nn.CrossEntropyLoss(), print_every=1, experiment_code = None, reset_weights = True, architecture = None):
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
            self.left_ear_layer = nn.Sequential(
                    nn.Linear(768, 20)
                            ).to(self.device)

            self.right_ear_layer = nn.Sequential(
                    nn.Linear(768, 20)
                    ).to(self.device)

            self.whole_image_layer = nn.Sequential(
                    nn.Linear(768, 23)
                    ).to(self.device)
        else:
            self.whole_image_layer = architecture[0].to(self.device)
            self.left_ear_layer = architecture[1].to(self.device)
            self.right_ear_layer = architecture[2].to(self.device)


        self.layer.load_state_dict(torch.load(Path(__file__).parent.parent / "weights/concept_w.pt", map_location=self.device, weights_only=False)) if (Path(__file__).parent.parent / "weights/concept_w.pt").exists() and reset_weights == False else None

        self.left_ear_optimizer = Adam(self.left_ear_layer.parameters(), lr)
        self.right_ear_optimizer = Adam(self.right_ear_layer.parameters(), lr)
        self.whole_image_optimizer = Adam(self.whole_image_layer.parameters(), lr)

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
            project="improve-concept-head-tunneled",

            # track hyperparameters and run metadata
            config={
                        "learning_rate": lr,
                        "code": experiment_code,
                        "loss" : loss
                    },
            dir = self.experiment_dir,
        settings=wandb.Settings(_disable_meta=True)
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
        if training:
            self.left_ear_layer.train()
            self.right_ear_layer.train()
            self.whole_image_layer.train()
        else:
            self.left_ear_layer.eval()
            self.right_ear_layer.eval()
            self.whole_image_layer.eval()


        total_loss = 0
        num_batches = len(loader)
        epoch_accuracies = {name: 0 for name in SEEK.attribute_names}
        epoch_accuracies.update({"average": 0, "whole_code": 0, "left_ear": 0, "right_ear": 0})

        for batch in loader:
            
            images, _, _, _, subject_SEEK_1h, ele_SEEK_1h, left_ear, right_ear, _, _, _ = batch
            
            labels_1h = ele_SEEK_1h.to(self.device) if predict_ele_SEEK else subject_SEEK_1h.to(self.device)
            
            images = images.to(self.device)

            embeddings = backbone.forward(images, left_ear, right_ear)
            
            whole_image_labels, left_ear_labels, right_ear_labels = SEEK.separate_one_hot(labels_1h)

            whole_image_embeddings = embeddings[:, :768]
            left_ear_embeddings = embeddings[:, 768:1536]
            right_ear_embeddings = embeddings[:, 1536:]

            left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
            right_ear_outputs = self.right_ear_layer(right_ear_embeddings)
            whole_image_outputs = self.whole_image_layer(whole_image_embeddings)

            w_loss = self.loss_fn(whole_image_outputs, whole_image_labels)
            l_loss = self.loss_fn(left_ear_outputs, left_ear_labels)
            r_loss = self.loss_fn(right_ear_outputs, right_ear_labels)

            outputs = SEEK.rescontruct_from_separated_prob_vectors(whole_image_outputs, left_ear_outputs, right_ear_outputs)

            total_loss += w_loss.item() + l_loss.item() + r_loss.item()

            batch_accuracies = self.accuracy_fn(outputs, labels_1h)

            for name, value in batch_accuracies.items():
                epoch_accuracies[name] += value

            if training:
                self.left_ear_optimizer.zero_grad()
                self.right_ear_optimizer.zero_grad()
                self.whole_image_optimizer.zero_grad()

                w_loss.backward()
                l_loss.backward()
                r_loss.backward()

                self.left_ear_optimizer.step()
                self.right_ear_optimizer.step()
                self.whole_image_optimizer.step()

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

        print(f"Training ConceptHead for {num_epochs} epochs, reseting weights, and the backbone parameters are,", any(param.requires_grad for param in backbone.layer.parameters()), ' concept head parameters are ', any(param.requires_grad for param in self.left_ear_layer.parameters()))

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
                best_weights = [self.whole_image_layer.state_dict(),self.left_ear_layer.state_dict() , self.right_ear_layer.state_dict()]

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

        self.whole_image_layer.load_state_dict(best_weights[0])
        self.left_ear_layer.load_state_dict(best_weights[1])
        self.right_ear_layer.load_state_dict(best_weights[2])

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
        
        #wandb.summary["test_loss"] = test_loss
        #wandb.summary["test_acc"] = test_acc["average"]
        #wandb.summary["test_acc_whole_code"] = test_acc["whole_code"]
        #wandb.summary["all_test_acc"] = test_acc

        print(f"ConceptHead - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['average'] * 100:.2f}% - Test whole-code: {test_acc['whole_code'] * 100:.2f}%")

        print(f"ConceptHead - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['average'] * 100:.2f}%")
        
        with open(self.experiment_dir / 'concept_head_test_accuracy.txt', 'w') as f:
            for name, value in test_acc.items():
                accuracy_str = f"{name} Accuracy: {value * 100:.2f}%"
                print(accuracy_str)
                f.write(accuracy_str + '\n')
        
        torch.save([self.whole_image_layer.state_dict(), self.left_ear_layer.state_dict(), self.right_ear_layer.state_dict()], self.experiment_dir / 'concept_w.pt')

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
        """Freezes all layers in the ConceptHead by disabling gradients."""
        for layer in [self.whole_image_layer, self.left_ear_layer, self.right_ear_layer]:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
            
    def unfreeze(self):
        """Unfreezes all layers in the ConceptHead for training."""
        for layer in [self.whole_image_layer, self.left_ear_layer, self.right_ear_layer]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    def collect_embeddings(self, loader, backbone, intervention_fn = None):
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        self.freeze()

        print('collecting concepts from concept head')
        for batch in tqdm(loader):
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[2].to(self.device), batch[4], batch[5], batch[6].to(self.device), batch[7].to(self.device)

            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)
                
                whole_image_embeddings = embeddings[:, :768]
                left_ear_embeddings = embeddings[:, 768:1536]
                right_ear_embeddings = embeddings[:, 1536:]

                left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
                right_ear_outputs = self.right_ear_layer(right_ear_embeddings)
                whole_image_outputs = self.whole_image_layer(whole_image_embeddings)

                outputs = SEEK.rescontruct_from_separated_prob_vectors(whole_image_outputs, left_ear_outputs, right_ear_outputs)

                predicted_SEEK = SEEK.closest_valid_one_hot(outputs)

                if intervention_fn is not None:
                    predicted_SEEK = intervention_fn(predicted_SEEK, subject_SEEK, ele_SEEK)
                
                predicted_SEEK = predicted_SEEK.to(self.device)

                collected_embeddings = torch.cat((collected_embeddings, predicted_SEEK))
                collected_labels = torch.cat((collected_labels, ele_id_label))

        return collected_embeddings, collected_labels



class MultiOutputNN(nn.Module):
    def __init__(self, input_size, lengths):
        """
        Multi-output neural network for categorical classification.

        Args:
        - input_size (int): Number of input features.
        - lengths (dict): Dictionary mapping category names to number of classes.
        """
        super(MultiOutputNN, self).__init__()
        
        self.shared_fc = nn.Linear(input_size, 512)  # Shared feature extractor
        self.relu = nn.ReLU()
        
        # Create independent output layers for each category
        self.output_heads = nn.ModuleDict({
            key: nn.Linear(512, num_classes) for key, num_classes in lengths.items()
        })

    def forward(self, x):
        x = self.relu(self.shared_fc(x))  # Shared representation

        # Compute logits for each category
        outputs = {key: head(x) for key, head in self.output_heads.items()}

        return outputs  # Dictionary of logits per category


def compute_multiclass_loss(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, requires_grad=True).to(targets.device)

    for idx, (key, logits) in enumerate(outputs.items()):
        category_targets = targets[:, idx]  # Ensure correct slicing
        loss = criterion(logits, category_targets)  
        total_loss = total_loss + loss  

    return total_loss

class ConceptHeadTunneledCategorical:
    def __init__(self, lr=0.001, loss=nn.CrossEntropyLoss(), print_every=1, experiment_code = None, reset_weights = True):
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
        self.print_every = print_every

        # Create dictionaries with attribute names and their corresponding lengths
        whole_lengths = {list(SEEK.lengths.keys())[i]: list(SEEK.lengths.values())[i] 
                for i in SEEK.whole_indices}
        left_lengths = {list(SEEK.lengths.keys())[i]: list(SEEK.lengths.values())[i] 
                for i in SEEK.left_indices}
        right_lengths = {list(SEEK.lengths.keys())[i]: list(SEEK.lengths.values())[i] 
                for i in SEEK.right_indices}
        

        self.whole_image_layer = MultiOutputNN(768, whole_lengths).to(self.device)
        self.left_ear_layer = MultiOutputNN(768, left_lengths).to(self.device)
        self.right_ear_layer = MultiOutputNN(768, right_lengths).to(self.device)

        self.left_ear_layer = nn.Sequential(
                nn.Linear(768, 20)
                        ).to(self.device)

        self.right_ear_layer = nn.Sequential(
                nn.Linear(768, 20)
                ).to(self.device)

        self.whole_image_layer = nn.Sequential(
                nn.Linear(768, 23)
                ).to(self.device)

        self.layer.load_state_dict(torch.load(Path(__file__).parent.parent / "weights/concept_w.pt", map_location=self.device, weights_only=False)) if (Path(__file__).parent.parent / "weights/concept_w.pt").exists() and reset_weights == False else None

        self.left_ear_optimizer = Adam(self.left_ear_layer.parameters(), lr)
        self.right_ear_optimizer = Adam(self.right_ear_layer.parameters(), lr)
        self.whole_image_optimizer = Adam(self.whole_image_layer.parameters(), lr)

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
            project="improve-concept-head-tunneled",

            # track hyperparameters and run metadata
            config={
                        "learning_rate": lr,
                        "code": experiment_code,
                        "loss" : loss
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
        print('redicted[0]', SEEK(predicted_one_hot[0]))
        print('labels[0]', SEEK(labels[0]))
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
        accuracies['whole_image'] = (accuracies['sex'] + accuracies['age'] + accuracies['right_tusk'] + accuracies['left_tusk'] + accuracies['right_extreme'] + accuracies['left_extreme'] + accuracies['ear_special'] + accuracies['body_special'])/8
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
        if training:
            self.left_ear_layer.train()
            self.right_ear_layer.train()
            self.whole_image_layer.train()
        else:
            self.left_ear_layer.eval()
            self.right_ear_layer.eval()
            self.whole_image_layer.eval()


        total_loss = 0
        num_batches = len(loader)
        epoch_accuracies = {name: 0 for name in SEEK.attribute_names}
        epoch_accuracies.update({"average": 0, "whole_code": 0, "left_ear": 0, "right_ear": 0, "whole_image": 0})

        for batch in loader:
            
            preprocessed_image, subject_id, ele_id_label, identified, subject_SEEK_1h, ele_SEEK_1h, left_ear, right_ear, subject_SEEK, ele_SEEK, idx = batch
            
            #labels_1h = ele_SEEK_1h.to(self.device) if predict_ele_SEEK else subject_SEEK_1h.to(self.device)
            labels = ele_SEEK if predict_ele_SEEK else subject_SEEK
            labels_1h = ele_SEEK_1h if predict_ele_SEEK else subject_SEEK_1h

            whole_labels_1h, left_labels_1h, right_labels_1h = SEEK.separate_one_hot(labels_1h)
            
            labels_cat, whole_cat, left_cat, right_cat = SEEK.batch_categorical(labels_1h.to(self.device)) # THAT WORKS

            #print('labels[0]', labels[0])
            #print('whole_cat[0]', whole_cat[0])
            #print('left_cat[0]', left_cat[0])
            #print('right_cat[0]', right_cat[0])
            
            
            images = preprocessed_image.to(self.device)

            embeddings = backbone.forward(images, left_ear, right_ear)
            
            #whole_image_labels, left_ear_labels, right_ear_labels = SEEK.separate_one_hot(labels_1h)

            whole_image_embeddings = embeddings[:, :768]
            left_ear_embeddings = embeddings[:, 768:1536]
            right_ear_embeddings = embeddings[:, 1536:]

            #print('whole_image_embeddings shape:', whole_image_embeddings.shape)
            #print('left_ear_embeddings shape:', left_ear_embeddings.shape)
            #print('right_ear_embeddings shape:', right_ear_embeddings.shape)

            whole_image_outputs = self.whole_image_layer(whole_image_embeddings) # That seem to work (not fully sure)
            left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
            right_ear_outputs = self.right_ear_layer(right_ear_embeddings)


            #concatenated_logits = torch.cat([logits for logits in whole_image_outputs.values()], dim=1)
            #print('outputs', left_ear_outputs)

            def concat_logits(outputs): # THIS METHODS WORKS FINE : left_ear_outputs is well transformed into left_logits
                concatenated_list = []

                for i in range(len(images)):  # Iterate over each individual in batch

                    individual_logits = torch.cat([logits[i] for logits in outputs.values()], dim=0)
                    concatenated_list.append(individual_logits)

                return torch.stack(concatenated_list, dim=0)
            
            #whole_logits = concat_logits(whole_image_outputs)
            #left_logits = concat_logits(left_ear_outputs)
            #right_logits = concat_logits(right_ear_outputs)

            whole_logits = whole_image_outputs
            left_logits = left_ear_outputs
            right_logits = right_ear_outputs
            

            #print('left_ear_logits', left_logits)

            #print(f"whole_image_outputs: {left_ear_outputs}")
            #print(f"whole_cat: {left_cat}")


            #w_loss = compute_multiclass_loss(whole_image_outputs, whole_cat) # WHY IS IT WHOLE IMAGE OUTPUT AND NOT LOGITS
            #l_loss = compute_multiclass_loss(left_ear_outputs, left_cat)
            #r_loss = compute_multiclass_loss(right_ear_outputs, right_cat)

            w_loss = F.mse_loss(whole_logits, whole_labels_1h.to(self.device))
            l_loss = F.mse_loss(left_logits, left_labels_1h.to(self.device))
            r_loss = F.mse_loss(right_logits, right_labels_1h.to(self.device))

            outputs = SEEK.rescontruct_from_separated_prob_vectors(whole_logits, left_logits, right_logits).to(self.device)

            total_loss += w_loss.item() + l_loss.item() + r_loss.item()
            #print('total_loss', total_loss)

            batch_accuracies = self.accuracy_fn(outputs, labels_1h.to(self.device))

            for name, value in batch_accuracies.items():
                epoch_accuracies[name] += value

            if training:
                self.left_ear_optimizer.zero_grad()
                self.right_ear_optimizer.zero_grad()
                self.whole_image_optimizer.zero_grad()

                w_loss.backward()
                l_loss.backward()
                r_loss.backward()

                self.left_ear_optimizer.step()
                self.right_ear_optimizer.step()
                self.whole_image_optimizer.step()

                for name, param in self.left_ear_layer.named_parameters():
                    if param.grad is None:
                        print(f"⚠️ WARNING: Gradient for {name} in left_ear_layer is None!")

                for name, param in self.right_ear_layer.named_parameters():
                    if param.grad is None:
                        print(f"⚠️ WARNING: Gradient for {name} in right_ear_layer is None!")

                for name, param in self.whole_image_layer.named_parameters():
                    if param.grad is None:
                        print(f"⚠️ WARNING: Gradient for {name} in whole_image_layer is None!")


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

        print(f"Training ConceptHead for {num_epochs} epochs, reseting weights, and the backbone parameters are,", any(param.requires_grad for param in backbone.layer.parameters()), ' concept head parameters are ', any(param.requires_grad for param in self.left_ear_layer.parameters()))

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
                best_weights = [self.whole_image_layer.state_dict(),self.left_ear_layer.state_dict() , self.right_ear_layer.state_dict()]

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

        self.whole_image_layer.load_state_dict(best_weights[0])
        self.left_ear_layer.load_state_dict(best_weights[1])
        self.right_ear_layer.load_state_dict(best_weights[2])

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
        
        #wandb.summary["test_loss"] = test_loss
        #wandb.summary["test_acc"] = test_acc["average"]
        #wandb.summary["test_acc_whole_code"] = test_acc["whole_code"]
        #wandb.summary["all_test_acc"] = test_acc

        print(f"ConceptHead - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['average'] * 100:.2f}% - Test whole-code: {test_acc['whole_code'] * 100:.2f}%")

        print(f"ConceptHead - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc['average'] * 100:.2f}%")
        
        with open(self.experiment_dir / 'concept_head_test_accuracy.txt', 'w') as f:
            for name, value in test_acc.items():
                accuracy_str = f"{name} Accuracy: {value * 100:.2f}%"
                print(accuracy_str)
                f.write(accuracy_str + '\n')
        
        torch.save([self.whole_image_layer.state_dict(), self.left_ear_layer.state_dict(), self.right_ear_layer.state_dict()], self.experiment_dir / 'concept_w.pt')

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
        """Freezes all layers in the ConceptHead by disabling gradients."""
        for layer in [self.whole_image_layer, self.left_ear_layer, self.right_ear_layer]:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
            
    def unfreeze(self):
        """Unfreezes all layers in the ConceptHead for training."""
        for layer in [self.whole_image_layer, self.left_ear_layer, self.right_ear_layer]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    def collect_embeddings(self, loader, backbone, intervention_fn = None):
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        self.freeze()

        print('collecting concepts from concept head')
        for batch in tqdm(loader):
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[2].to(self.device), batch[4], batch[5], batch[6].to(self.device), batch[7].to(self.device)

            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)
                
                whole_image_embeddings = embeddings[:, :768]
                left_ear_embeddings = embeddings[:, 768:1536]
                right_ear_embeddings = embeddings[:, 1536:]

                left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
                right_ear_outputs = self.right_ear_layer(right_ear_embeddings)
                whole_image_outputs = self.whole_image_layer(whole_image_embeddings)

                outputs = SEEK.rescontruct_from_separated_prob_vectors(whole_image_outputs, left_ear_outputs, right_ear_outputs)

                predicted_SEEK = SEEK.closest_valid_one_hot(outputs)

                if intervention_fn is not None:
                    predicted_SEEK = intervention_fn(predicted_SEEK, subject_SEEK, ele_SEEK)
                
                predicted_SEEK = predicted_SEEK.to(self.device)

                collected_embeddings = torch.cat((collected_embeddings, predicted_SEEK))
                collected_labels = torch.cat((collected_labels, ele_id_label))

        return collected_embeddings, collected_labels
