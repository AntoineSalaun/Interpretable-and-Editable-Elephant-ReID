from datetime import datetime
import pathlib as Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, num_classes, criterion = nn.CrossEntropyLoss(), lr=0.001, experiment_code = None):
        
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # layer 
        self.layer = nn.Sequential(
            nn.Linear(2304, num_classes),
            nn.Softmax()
        ).to(self.device)

        # Load default weights
        if (Path.Path.cwd().parent / 'weights' / 'classifier_w.pt').exists():
            self.layer.load_state_dict(torch.load(Path.Path.cwd().parent / 'weights' / 'classifier_w.pt', map_location=self.device))
        else: print('weights not found')


        self.loss_fn = criterion
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)
        
        exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = Path.Path.cwd().parent / 'experiments' / f'exp_{exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.experiment_dir / 'note.txt', 'a') as f: 
            f.write('---classifier---\n')
    
    def compute_accuracy(self, logits, labels):
        """Compute accuracy given logits and true labels."""
        _, predicted = torch.max(logits, dim=1)  # Get predicted class indices
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total  # Fraction of correctly predicted samples
        return accuracy 



    def epoch_pass(self, loader, backbone, training = True):
    
        self.layer.train() if training else self.layer.eval()

        total_loss, total_accuracy, total_batches = 0,0,0

        for batch in loader:
            images, ele_id_label, subject_SEEK, left_ears, right_ears = batch[0].to(self.device),  batch[2].to(self.device), batch[4], batch[6].to(self.device), batch[7].to(self.device)
            #print(ele_id_label)
            
            embeddings = backbone.forward(images, left_ears, right_ears)
            
            # Forward pass
            logits = self.layer(embeddings)
            
            # Loss with integer labels
            loss = self.loss_fn(logits, ele_id_label)  
            total_loss += loss.item()

            # Compute metrics
            total_accuracy += self.compute_accuracy(logits, ele_id_label)
            total_batches += 1

            # Backward pass and optimization
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # accumulate loss and accuracy
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = total_accuracy / total_batches

        return epoch_loss, epoch_accuracy



    def train(self, train_loader, val_loader, backbone, num_epochs = 30):

        # Initialize weights for Linear layers
        for m in self.layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Freeze the concept head
        
        print('Training the classifier : Backbone parameters require gradients:', any([param.requires_grad for param in backbone.layer.parameters()]), '\nClassifier parameters require gradients:', any([param.requires_grad for param in self.layer.parameters()]))

        history = []
        best_val_accuracy = 0


        for epoch in range(num_epochs):
            train_loss, train_acc = self.epoch_pass(train_loader, backbone, training=True)
            val_loss, val_acc = self.epoch_pass(val_loader, backbone, training=False)

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc_avg": train_acc*100, "val_acc_avg": val_acc, "train_acc_whole_code": train_acc*100, "val_acc_whole_code": val_acc})

            # Save best model weights
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_weights = self.layer.state_dict()

            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}% - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")
        
        # Save best weights
        torch.save(best_weights, self.experiment_dir / 'classifier_w.pt')

        # Convert history to a DataFrame and save as CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / 'classifier_history.csv', index=False)

    def test(self, test_loader, backbone):
        self.layer.eval()
        test_loss, test_acc = self.epoch_pass(test_loader, backbone, training=False)

        print(f"Classifier - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

        with open(self.experiment_dir / 'classifier_test_accuracy.txt', 'w') as f:
            f.write(f"{test_acc:.2f}")

    def freeze(self):
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.layer.train()
        for param in self.layer.parameters():
            param.requires_grad = True
        
