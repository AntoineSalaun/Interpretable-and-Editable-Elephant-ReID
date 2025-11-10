import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from retrieval import Retrieval
from seek_code import SEEK
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def categorical_CE_loss(output,labels):

    idx = 0
    att_loss = {}
    tot_loss = 0
    
    for key, value in SEEK.lengths.items():
        att_output = output[:, idx:idx + value]
        att_labels = labels[:, idx:idx + value]

        att_loss[key] = F.cross_entropy(att_output, att_labels)
        tot_loss += att_loss[key]
        idx += value


    return tot_loss


class ConceptHeadTunneled:
    def __init__(self, lr=1e-5, loss=categorical_CE_loss,  experiment_code = None, reset_weights = True, layer = None, pretraining = 'concept_w'):

        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.input_dim = 768 * 3  # Handle concatenated embeddings
        self.layer = layer

        if (Path(__file__).parent.parent / f"weights/{pretraining}.pt").exists() and reset_weights == False:
            self.layer.load_model(Path(__file__).parent.parent / f"weights/{pretraining}.pt")
        
        #self.optimizer = Adam(self.layer.parameters(), lr)

        self.loss_fn = loss
        self.lr = lr

        # Create experiment directory
        exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = Path(__file__).parent.parent / 'experiments' / f'exp_{exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.exp_code = exp_code


    def accuracy_fn(self, predicted, labels):
        """
        Computes accuracy for each attribute and the whole SEEK code.
        """
        predicted_one_hot = SEEK.closest_valid_one_hot(predicted)
        accuracies = {name: 0 for name in SEEK.attribute_names + ['big_errors']}

        index = 0
        for name in SEEK.attribute_names:
            length = SEEK.lengths[name]
            pred_slice = predicted_one_hot[:, index:index + length]
            label_slice = labels[:, index:index + length]
            pred_indices = torch.argmax(pred_slice, dim=1)
            label_indices = torch.argmax(label_slice, dim=1)
            accuracies[name] = (pred_indices == label_indices).float().mean().item()
            
            #print('pred_indices', pred_indices.shape)
            if pred_indices.shape[0] == 1:
                if SEEK.mappings[name][pred_indices] != SEEK.mappings[name][label_indices]:
                    if SEEK.mappings[name][pred_indices] == '_' or SEEK.mappings[name][label_indices] == '_':
                        accuracies['big_errors'] += 0
                    else:
                        accuracies['big_errors'] += 1

            index += length

        correct_whole_code = torch.all(predicted_one_hot == labels, dim=1).float().mean().item()

        accuracies['average'] = sum(accuracies.values()) / len(SEEK.attribute_names)
        accuracies['whole_code'] = correct_whole_code
        accuracies['left_ear'] = (accuracies['L_tear_1'] + accuracies['L_hole_1'] + accuracies['L_tear_2'] + accuracies['L_hole_2'] + accuracies['left_extreme'])/5
        accuracies['right_ear'] = (accuracies['R_tear_1'] + accuracies['R_hole_1'] + accuracies['R_tear_2'] + accuracies['R_hole_2'] + accuracies['right_extreme'])/5

        return accuracies

    def epoch_pass(self, loader, backbone, training=True, predict_ele_SEEK = False):

        if training:
            backbone.layer.train()
            self.layer.train()
        else:
            backbone.layer.eval()
            self.layer.eval()


        epoch_loss, images_seen = 0, 0
        epoch_accuracies = {name: 0 for name in SEEK.attribute_names}
        epoch_accuracies.update({"average": 0, "whole_code": 0, "left_ear": 0, "right_ear": 0, "big_errors": 0})

        for batch in loader:
            
            images, _, _, _, subject_SEEK_1h, ele_SEEK_1h, left_ear, right_ear, _, _, idx = batch
            
            labels_1h = ele_SEEK_1h.to(self.device) if predict_ele_SEEK else subject_SEEK_1h.to(self.device)
            
            images = images.to(self.device)

            embeddings = backbone.forward(images, left_ear, right_ear)
            
            outputs = self.layer(embeddings)

            loss = self.loss_fn(outputs, labels_1h)
            batch_accuracies = self.accuracy_fn(outputs, labels_1h)

            epoch_loss += loss.item() * len(batch)
            for name, value in batch_accuracies.items(): epoch_accuracies[name] += value * len(batch)
            images_seen += len(batch)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        epoch_loss = epoch_loss / images_seen
        epoch_accuracies = {name: accuracy / images_seen for name, accuracy in epoch_accuracies.items()}
        return epoch_loss, epoch_accuracies


    def train(self, train_loader, val_loader, backbone, num_epochs=10, predict_ele_SEEK = False):
        
        self.optimizer = Adam(list(self.layer.parameters()) + list(backbone.layer.parameters()), lr=self.lr, weight_decay=0.1*self.lr)
        print(f"Training ConceptHead for {num_epochs} epochs, reseting weights, and the backbone parameters are,", any(param.requires_grad for param in backbone.layer.parameters()), ' concept head parameters are ', any(param.requires_grad for param in self.layer.parameters()))

        history = []
        for epoch in range(num_epochs):
            # Perform a training and validation pass
            train_loss, train_acc = self.epoch_pass(train_loader, backbone, training=True, predict_ele_SEEK = predict_ele_SEEK)
            val_loss, val_acc = self.epoch_pass(val_loader, backbone, training=False, predict_ele_SEEK = predict_ele_SEEK)

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "train_acc_avg": train_acc["average"], "val_acc_avg": val_acc["average"], "train_acc_whole_code": train_acc["whole_code"], "val_acc_whole_code": val_acc["whole_code"]})
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - Train Acc: {train_acc['average'] * 100:.2f}%, Val Acc: {val_acc['average'] * 100:.2f}% - Train whole-code: {train_acc['whole_code'] * 100:.2f}%, Val whole-code: {val_acc['whole_code'] * 100:.2f}% - Train left ear: {train_acc['left_ear'] * 100:.2f}%, Val left ear: {val_acc['left_ear'] * 100:.2f}% - Train right ear: {train_acc['right_ear'] * 100:.2f}%, Val right ear: {val_acc['right_ear'] * 100:.2f}%")

        # Save best weights
        self.layer.save_model(self.experiment_dir / 'concept_w.pt')
        torch.save(backbone.layer.state_dict(), self.experiment_dir / 'backbone_for_concepts_w.pt')

        # Convert history to a DataFrame and save as CSV
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / 'concept_training_history.csv', index=False)

        return history_df

    def test(self, test_loader, backbone, predict_ele_SEEK = False):
        print("--------------TEST---------------")
        test_loss, test_acc = self.epoch_pass(test_loader, backbone, training=False, predict_ele_SEEK = predict_ele_SEEK)
        
        with open(self.experiment_dir / 'concept_head_test_accuracy.txt', 'w') as f:
            for name, value in test_acc.items():
                accuracy_str = f"{name} Accuracy: {value * 100:.2f}%"
                print(accuracy_str)
                f.write(accuracy_str + '\n')

        return test_loss, test_acc

    def retrieval_test(self, backbone_for_concepts, train_loader, test_loader, intervention_fns):
        """
        Test the model with different intervention functions and store results as summaries.
        """

        with open(self.experiment_dir / 'SEEK_retrieval_results.txt', 'w') as f:
            f.write(f'Testing with intervention functions: {", ".join(intervention_fns.keys())}\n')

        results = {}

        retrieval = Retrieval(experiment_code=self.exp_code)
    
        distances = ['cosine_sim', 'seek_homemade', 'seek_homemade_2', 'seek_homemade_3']
        for distance in distances:
            print(f"\nTesting with {distance} distance:")
            for name, fn in intervention_fns.items():
                print(f'Retrieval with {name} correction-------------------------------------')
                metrics = retrieval.evaluate_model(
                    model=self,
                    train_loader=train_loader,
                    test_loader=test_loader, model_to_evaluate='concept_head',
                    ba=None,
                    ch=None,
                    backbone_for_concepts=backbone_for_concepts,
                    intervention_fn=fn,
                    show_matches=False, show_plot=False, show_tsne=False, print_results=False, aggregate_seeks=False,
                    distance=distance
                )
                
                results[f"{name}_{distance}"] = metrics
            
                print(f"{name} - " + ", ".join([f"Recall@{k}: {v * 100:.2f}%" for k, v in metrics.items()]))

        return results

    def freeze(self):
        """Freezes all layers in the ConceptHead by disabling gradients."""
        self.layer.eval()
        for param in self.layer.parameters():
            param.requires_grad = False

            
    def unfreeze(self):
        """Unfreezes all layers in the ConceptHead for training."""
        for layer in [self.layer.whole_image_layer, self.layer.left_ear_layer, self.layer.right_ear_layer]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True


    def collect_embeddings(self, loader, backbone, intervention_fn = None, aggregate_seeks = False, aggregate_rules = None):
        
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        self.freeze()

        for batch in tqdm(loader):
            #preprocessed_image, subject_id, ele_id_label, identified, subject_SEEK_1hot, ele_SEEK_1hot, left_ear, right_ear, subject_SEEK, ele_SEEK, idx
            images, ele_id_label, subject_SEEK, ele_SEEK, left_ears, right_ears = batch[0].to(self.device), batch[2].to(self.device), batch[4], batch[5], batch[6].to(self.device), batch[7].to(self.device)

            with torch.no_grad():
                
                if intervention_fn is SEEK.perfect_correction or intervention_fn is SEEK.oracle_correction: # If we will correct either perfectly or with oracle, no need to compute initial prediction
                    embeddings = None
                    outputs = None
                    predicted_SEEK = None
                else: # Regular scenario, compute embeddings and predictions
                    embeddings = backbone.forward(images, left_ears, right_ears)
                    outputs = self.layer(embeddings)
                    predicted_SEEK = SEEK.closest_valid_one_hot(outputs)

                if intervention_fn is not None:
                    predicted_SEEK = intervention_fn(predicted_SEEK, subject_SEEK, ele_SEEK)
                
                collected_embeddings = torch.cat((collected_embeddings, predicted_SEEK.to(self.device)))
                collected_labels = torch.cat((collected_labels, ele_id_label))
        
        if aggregate_seeks:
            subject_seek, collected_embeddings, collected_labels = SEEK.aggregate_seek(collected_embeddings, collected_labels, rules = aggregate_rules)
            
        collected_embeddings = collected_embeddings.cpu()
        collected_labels = collected_labels

        return collected_embeddings, collected_labels

   


class ThreeHeadNN(nn.Module):
    def __init__(self):
        super(ThreeHeadNN, self).__init__()

        self.whole_image_layer = nn.Sequential(
                nn.Linear(768, 23)
                ).to('cuda')
        self.left_ear_layer = nn.Sequential(
                nn.Linear(768, 20)
                        ).to('cuda')

        self.right_ear_layer = nn.Sequential(
                nn.Linear(768, 20)
                ).to('cuda')

    def forward(self, embeddings):
        whole_image_embeddings = embeddings[:, :768]
        left_ear_embeddings = embeddings[:, 768:1536]
        right_ear_embeddings = embeddings[:, 1536:]

        left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
        right_ear_outputs = self.right_ear_layer(right_ear_embeddings)
        whole_image_outputs = self.whole_image_layer(whole_image_embeddings)

        outputs = SEEK.rescontruct_from_separated_prob_vectors(whole_image_outputs, left_ear_outputs, right_ear_outputs)

        return outputs  # No softmax, CrossEntropyLoss applies it
    

class MultiHeadNN(nn.Module):
    def __init__(self):
        super(MultiHeadNN, self).__init__()

        self.whole_image_layer = nn.Sequential(
                nn.Linear(768, 768),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(768, 256)
                ).to('cuda')
        
        self.left_ear_layer = nn.Sequential(
                nn.Linear(768, 768),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(768, 256)
                ).to('cuda')
        
        self.right_ear_layer = nn.Sequential(
                nn.Linear(768, 768),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(768, 256)
                ).to('cuda')
        
        self.heads = {}        
        for key, value in SEEK.lengths.items():
                self.heads[key] = nn.Sequential(
                        nn.Linear(256, 128),
                        nn.LeakyReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(128, value)
                        ).to('cuda')
        

    def forward(self, embeddings):
        whole_image_embeddings = embeddings[:, :768]
        left_ear_embeddings = embeddings[:, 768:1536]
        right_ear_embeddings = embeddings[:, 1536:]

        left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
        right_ear_outputs = self.right_ear_layer(right_ear_embeddings)
        whole_image_outputs = self.whole_image_layer(whole_image_embeddings)

        outputs = {}
        output_tensor = torch.zeros(embeddings.shape[0], 63).to('cuda')
        idx = 0
        for key, value in SEEK.lengths.items():
                if key in ['sex', 'age', 'right_tusk', 'left_tusk', 'right_extreme', 'left_extreme', 'ear_special', 'body_special']:
                        outputs[key] = self.heads[key](whole_image_outputs)
                elif key in ['R_tear_1', 'R_hole_1', 'R_tear_2', 'R_hole_2']:
                        outputs[key] = self.heads[key](left_ear_outputs)
                elif key in ['L_tear_1', 'L_hole_1', 'L_tear_2', 'L_hole_2']:
                        outputs[key] = self.heads[key](right_ear_outputs)
                else:
                       raise ValueError(f"Unknown key: {key}")
                output_tensor[:, idx:idx + value] = outputs[key]
                idx += value

        return output_tensor  # No softmax, CrossEntropyLoss applies it
    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model state dict."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        print(f"Model loaded from {path}")


class CrossedHeadNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.whole_image_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 256)
            ).to('cuda')

        self.left_ear_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 256)
            ).to('cuda')

        self.right_ear_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 256)
            ).to('cuda')

        self.heads = nn.ModuleDict()        
        for key, value in SEEK.lengths.items():
            if key in ['sex', 'age', 'right_tusk', 'left_tusk', 'right_extreme', 'left_extreme', 'ear_special', 'body_special']: 
                self.heads[key] = nn.Linear(256, value).to('cuda')
            elif key in ['R_tear_1', 'R_hole_1', 'R_tear_2', 'R_hole_2', 'L_tear_1', 'L_hole_1', 'L_tear_2', 'L_hole_2']:
                self.heads[key] = nn.Linear(512, value).to('cuda')
            else:
                raise ValueError(f"Unknown key: {key}")

    def forward(self, embeddings):
        whole_image_embeddings = embeddings[:, :768]
        left_ear_embeddings = embeddings[:, 768:1536]
        right_ear_embeddings = embeddings[:, 1536:]

        left_ear_outputs = self.left_ear_layer(left_ear_embeddings)
        right_ear_outputs = self.right_ear_layer(right_ear_embeddings)
        whole_image_outputs = self.whole_image_layer(whole_image_embeddings)

        whole_and_left = torch.cat((whole_image_outputs, left_ear_outputs), dim=1)
        whole_and_right = torch.cat((whole_image_outputs, right_ear_outputs), dim=1)

        outputs = {}
        output_tensor = torch.zeros(embeddings.shape[0], 63).to('cuda')
        idx = 0
        for key, value in SEEK.lengths.items():
            if key in ['sex', 'age', 'right_tusk', 'left_tusk', 'right_extreme', 'left_extreme', 'ear_special', 'body_special']:
                outputs[key] = self.heads[key](whole_image_outputs)
            elif key in ['R_tear_1', 'R_hole_1', 'R_tear_2', 'R_hole_2']:
                outputs[key] = self.heads[key](whole_and_left)
            elif key in ['L_tear_1', 'L_hole_1', 'L_tear_2', 'L_hole_2']:
                outputs[key] = self.heads[key](whole_and_right)
            else:
                raise ValueError(f"Unknown key: {key}")
            
            output_tensor[:, idx:idx + value] = outputs[key]
            idx += value

        return output_tensor  # No softmax, CrossEntropyLoss applies it
    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model state dict."""
        self.load_state_dict(torch.load(path, map_location='cuda'))
        self.to('cuda')
        print(f"Concept head loaded from {path}")