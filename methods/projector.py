import pandas as pd
from datetime import datetime
import pathlib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from seek_code import SEEK

class Projector(nn.Module):
    def __init__(self, criterion = nn.CrossEntropyLoss() , lr = 1e-3, experiment_code = None):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layer =    nn.Sequential(
                        nn.Linear(63, 512),
                        nn.LeakyReLU(),
                        nn.Linear(512, 2304),
                        nn.LeakyReLU()
        ).to(self.device)

        self.loss_fn = criterion
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)


        exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = pathlib.Path.cwd().parent / 'experiments' / f'exp_{exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.experiment_dir / 'note.txt', 'a') as f: 
            f.write('---projector---')

    def compute_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=1)
        correct = torch.sum(predictions == labels).item()
        return correct / len(labels)

    def epoch_pass(self, loader, ba, ch, cl, training=True, intervention_fn = None):

        # Are we training ?
        self.layer.train() if training else self.layer.eval()

        total_loss, total_accuracy, total_batches = 0, 0, 0
        

        
        for batch in loader:
            images, ele_id_label, subject_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[6].to('cuda'), batch[7].to('cuda')

            embeddings = ba.forward(images, left_ears, right_ears)   

            with torch.no_grad():

                # Compute the concepts from the backbone embeddings
                concept_logits = ch.layer(embeddings)
                hard_predicted_concepts = SEEK.closest_valid_one_hot(concept_logits)
                
                # Allow for intervention
                if intervention_fn is not None:
                    #print('We apply an intervention function')
                    edited_concepts = intervention_fn(hard_predicted_concepts, subject_SEEK)
                else: 
                    #print('No intervention function applied, edited_concepts = hard_predicted_concepts')
                    edited_concepts = hard_predicted_concepts

            # Projecting the concepts back to the embeddings space
            projected_concepts = self.layer(edited_concepts)
            # I want to do this > loss = criterion(projected_concepts, embeddings)

            # We add the concepts projected back to the intiial embeddings space
            edited_embeddings = projected_concepts + embeddings

            # Now we can pass the edited embeddings to the classifier
            logits = cl.layer(edited_embeddings)

            loss = self.loss_fn(logits, ele_id_label)

            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute metrics
            total_accuracy += self.compute_accuracy(logits, ele_id_label)
            total_batches += 1
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_accuracy / total_batches

        return epoch_loss, epoch_accuracy


    def train(self, train_loader, val_loader,  backbone , concept_head, classifier, num_epochs=10, intervention_fn = None, reset_wegihts = True):
        
        print('training projector- ', 'weights are reset' if reset_wegihts else 'weights are not reset')
        # Initialize projector weights
        if reset_wegihts:
            for layer in self.layer:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            
        print('Concept head parameters require gradients:', any(param.requires_grad for param in concept_head.layer.parameters()), '\nClassifier parameters require gradients:', any(param.requires_grad for param in classifier.layer.parameters()), '\nBackbone parameters require gradients:', any(param.requires_grad for param in backbone.layer.parameters()), '\nProjector parameters require gradients:', any(param.requires_grad for param in self.layer.parameters()))

        history = []
        best_val_accuracy = 0


        for epoch in range(num_epochs):
            train_loss, train_acc = self.epoch_pass(train_loader, backbone, ch=concept_head, cl=classifier, training=True, intervention_fn = intervention_fn)
            val_loss, val_acc = self.epoch_pass(val_loader, backbone, ch=concept_head, cl=classifier, training=False, intervention_fn = intervention_fn)

                        # Save best model weights
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_weights = self.layer.state_dict()

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc_avg": train_acc*100, "val_acc_avg": val_acc*100})
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}% - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")

        # Save best weights
        torch.save(best_weights, self.experiment_dir / 'projector_w.pt')

        # Convert history to a DataFrame and save as CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / 'projector_training_history.csv', index=False)

    def test(self, test_loader, backbone, classifier, concept_head, intervention_fn = None):

        test_loss, test_accuracy = self.epoch_pass(test_loader, backbone, ch=concept_head, cl=classifier, training=False, intervention_fn = intervention_fn)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")

        with open(self.experiment_dir / 'projector_test_accuracy.txt', 'w') as f: f.write(f"{test_accuracy*100:.2f}")

    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True
        

