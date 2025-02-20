import torch
from tqdm import tqdm
from seek_code import SEEK
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import numpy as np
import random
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

class Retrieval:
    def __init__(self, experiment_code = None):
        
        exp_code = experiment_code or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir = pathlib.Path.cwd().parent / 'experiments' / f'exp_{exp_code}'
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.experiment_dir / 'note.txt', 'a') as f: 
            f.write('---projector---')

    def collect_edited_embeddings(self, loader, backbone, concept_head, projector, intervention_fn =None):
        print('collecting the embeddings')
        backbone.freeze()
        concept_head.freeze()
        projector.freeze()
        
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        
        for batch in tqdm(loader):
            images, ele_id_label, subject_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[6].to('cuda'), batch[7].to('cuda')
            
            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)   
                # Compute the concepts from the backbone embeddings
                concept_logits = concept_head.layer(embeddings)
                hard_predicted_concepts = SEEK.closest_valid_one_hot(concept_logits)
                
                # Allow for intervention ???
                if intervention_fn is not None:
                    #print('We apply an intervention function')
                    edited_concepts = intervention_fn(hard_predicted_concepts, subject_SEEK)
                else: 
                    #print('No intervention function applied, edited_concepts = hard_predicted_concepts')
                    edited_concepts = hard_predicted_concepts

                # Projecting the concepts back to the embeddings space
                projected_concepts = projector.layer(edited_concepts)
                    
                # We add the concepts projected back to the intiial embeddings space
                edited_embeddings = projected_concepts + embeddings

                collected_embeddings = torch.cat((collected_embeddings, edited_embeddings))
                collected_labels = torch.cat((collected_labels, ele_id_label))
        
        return collected_embeddings, collected_labels

    def collect_MD_embeddings(self, loader, backbone, intervention_fn =None):

        backbone.freeze()
        
        collected_embeddings = torch.tensor([]).to('cuda')
        collected_labels = torch.tensor([]).to('cuda')

        
        for batch in tqdm(loader):
            images, ele_id_label, subject_SEEK, left_ears, right_ears = batch[0].to('cuda'),  batch[2].to('cuda'), batch[4], batch[6].to('cuda'), batch[7].to('cuda')
            
            with torch.no_grad():
                embeddings = backbone.forward(images, left_ears, right_ears)   
                # Compute the concepts from the backbone embeddings

                collected_embeddings = torch.cat((collected_embeddings, embeddings))
                collected_labels = torch.cat((collected_labels, ele_id_label))
        
        return collected_embeddings, collected_labels

    def cosine_similarity_matrix(self, query_embeddings, gallery_embeddings = None):
        """
        Compute the cosine similarity matrix between query and gallery embeddings.

        Args:
            query_embeddings (torch.Tensor): Tensor of query embeddings (num_queries, embedding_dim).
            gallery_embeddings (torch.Tensor): Tensor of gallery embeddings (num_gallery, embedding_dim).

        Returns:
            torch.Tensor: Similarity matrix (num_queries, num_gallery).
        """
        square = False
        if gallery_embeddings is None: 
            square = True
            gallery_embeddings = query_embeddings
        # Normalize embeddings for cosine similarity
        query_embeddings = F.normalize(query_embeddings, dim=1)
        gallery_embeddings = F.normalize(gallery_embeddings, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(query_embeddings, gallery_embeddings.T)

        if square:
            # Exclude self-similarity if only one set of embeddings is provided
            similarity_matrix.fill_diagonal_(0)

        return similarity_matrix

    def compute_recall_at_k(self, similarity_matrix, query_labels, gallery_labels, k=1):
        """
        Compute Recall@k for a retrieval task.

        Args:
            similarity_matrix (torch.Tensor): Similarity matrix (num_queries, num_gallery).
            query_labels (torch.Tensor): Labels for query embeddings (num_queries,).
            gallery_labels (torch.Tensor): Labels for gallery embeddings (num_gallery,).
            k (int): Number of top results to consider.

        Returns:
            float: Recall@k as a percentage.
        """
        # Get the indices of the top-k gallery items for each query
        top_k_indices = torch.topk(similarity_matrix, k=k, dim=1, largest=True).indices

        # Count the number of relevant items in the top-k results for each query
        total_relevant = 0
        for i, query_label in enumerate(query_labels):
            top_k_labels = gallery_labels[top_k_indices[i]]
            # Count if the query_label appears in the top-k labels
            if query_label in top_k_labels:
                total_relevant += 1
        
        # Calculate Recall@k
        recall_at_k = total_relevant / len(query_labels)
        return recall_at_k

    def plot_accuracy_vs_samples(self, similarity_matrix, query_labels, gallery_labels, ks):

        """Compute per-individual accuracy for different ks and prepare graph data."""
        count_acc = {k: {} for k in ks}
        count_n = {k: {} for k in ks}

        for k in ks:
            scores, idx = torch.topk(similarity_matrix.cpu(), k=k, dim=1, largest=True)
            topk = gallery_labels[idx]
            matches = (topk == query_labels.unsqueeze(1)).sum(dim=1)

            unique, counts = torch.unique(gallery_labels, return_counts=True)
            for u, c in zip(unique, counts):
                m = matches[query_labels == u].float()
                if len(m) > 0:
                    if c.item() not in count_acc[k]:
                        count_acc[k][c.item()] = []
                    if c.item() not in count_n[k]:
                        count_n[k][c.item()] = 0
                    count_acc[k][c.item()].append(torch.mean(m).cpu())
                    count_n[k][c.item()] += 1

        """Plot accuracy vs. number of samples per individual with a histogram overlay."""
        fig, ax1 = plt.subplots()

        # Plot accuracy
        for k in ks:
            x = sorted(count_acc[k].keys())
            y = [np.mean(count_acc[k][i]) for i in x]
            ax1.plot(x, y, label=f"Top-{k}")
        ax1.set_xlabel("# Samples per Individual in Database")
        ax1.set_ylabel("Average Accuracy")
        ax1.legend()

        # Overlay histogram
        ax2 = ax1.twinx()
        ax2.bar(count_n[ks[0]].keys(), count_n[ks[0]].values(), color="tab:grey", alpha=0.25)
        ax2.set_ylabel("# Individuals", color="tab:grey")
        ax2.tick_params(axis="y", labelcolor="tab:grey")

        plt.savefig(self.experiment_dir / 'accuracy_vs_samples.png')
        plt.show() 

    def evaluate_model(self, model, train_loader, test_loader, ba = None, ch = None, show_tsne = True, show_plot = True, show_matches = True, intervention_fn = None):

        ks = [1, 5, 10, 20, 100]

        model.freeze()

        with torch.no_grad():
            # Train
            if ba is None and ch is None: # evaluating backbone
                print('evaluating backbone')
                gallery_embeddings, gallery_labels = model.collect_embeddings(train_loader)
                query_embeddings, query_labels = model.collect_embeddings(test_loader)
            elif ba is not None and ch is None: # evaluating concept head
                print('evaluating concept head ', ' with intervention' if intervention_fn is not None else '')
                gallery_embeddings, gallery_labels = model.collect_embeddings(train_loader, ba, intervention_fn = intervention_fn)
                query_embeddings, query_labels = model.collect_embeddings(test_loader, ba, intervention_fn = intervention_fn)
            else: # evaluating projector
                print('evaluating projector ', ' with intervention' if intervention_fn is not None else '')
                gallery_embeddings, gallery_labels = model.collect_embeddings(train_loader, ba, ch, intervention_fn = intervention_fn)
                query_embeddings, query_labels = model.collect_embeddings(test_loader, ba, ch, intervention_fn = intervention_fn)
            
            gallery_sim_matrix = self.cosine_similarity_matrix(gallery_embeddings)
            train_recalls = {k: self.compute_recall_at_k(gallery_sim_matrix, gallery_labels, gallery_labels, k=k) for k in ks}

            # Test with new queries
            
            test_similarity_matrix = self.cosine_similarity_matrix(query_embeddings, gallery_embeddings)

            # Compute Recall@k
            test_recalls = {k: self.compute_recall_at_k(test_similarity_matrix, query_labels, gallery_labels, k=k) for k in ks}
            
            print("Train Recalls:")
            for k in ks:
                print(f"Recall@{k}: {train_recalls[k]*100:.2f}%")
            
            print("Test Recalls:")
            for k in ks:
                print(f"Recall@{k}: {test_recalls[k]*100:.2f}%")
            with open(self.experiment_dir / 'recalls.txt', 'w') as f: f.write(f"Train - Recall@{ks}: {train_recalls}\nTest - Recall@{ks}: {test_recalls}\n")
                        
            if show_tsne: self.make_tsne(gallery_embeddings, gallery_labels)

            # Plot accuracy vs. number of samples per individual
            if show_plot: self.plot_accuracy_vs_samples(test_similarity_matrix, query_labels, gallery_labels, ks)

            # Visualize retrieval results for queries with unique labels
            # train_loader.dataset.dataset, test_loader.dataset.dataset give access to the whole dataset (twice .dataset in orde to accest the Subset's whole dataset)
            if show_matches: self.visualize_matches(test_similarity_matrix, query_labels, gallery_labels, train_loader.dataset.dataset, train_loader.dataset.dataset, train_loader.dataset.dataset)


    def visualize_matches(self, similarity_matrix, query_labels, gallery_labels, retrieval_dataset, query_subset, gallery_subset, n_vis=4, k=600):
        """
        Visualize retrieval results for queries with unique labels.
        
        Args:
            similarity_matrix (torch.Tensor): Precomputed similarity matrix (queries x galleries).
            query_labels (torch.Tensor): Labels for the query set.
            gallery_labels (torch.Tensor): Labels for the gallery set.
            retrieval_dataset: Dataset object with `get_original_image(subject_id)` method.
            query_subset (list): Subset of the query dataset (e.g., indices or tuples with IDs).
            gallery_subset (list): Subset of the gallery dataset (e.g., indices or tuples with IDs).
            n_vis (int): Number of queries to visualize. Default is 4.
            k (int): Number of top-k gallery matches to consider. Default is 600.
        """
        # Step 1: Select random queries with unique true labels
        unique_labels = set()
        indices_by_label = {}

        for i, label in enumerate(query_labels):
            label = label.item()
            if label not in indices_by_label:
                indices_by_label[label] = []
            indices_by_label[label].append(i)

        selected_indices = []
        while len(selected_indices) < n_vis and indices_by_label:
            label = random.choice(list(indices_by_label.keys()))
            idx = random.choice(indices_by_label[label])
            selected_indices.append(idx)
            del indices_by_label[label]

        # Step 2: Perform top-k retrieval
        scores, idx = torch.topk(similarity_matrix, k=k, dim=1)

        # Step 3: Load samples for visualization
        samples = []
        for i in selected_indices:
            query_id = query_subset[i][1]
            seen_indices = set()
            top_1_match = None
            most_confident_correct = None
            least_confident_correct = None

            for k_idx in range(k):
                gallery_idx = idx[i][k_idx].item()
                if gallery_idx in seen_indices:
                    continue
                seen_indices.add(gallery_idx)

                gallery_id = gallery_subset[gallery_idx][1]
                correct = query_labels[i].item() == gallery_labels[gallery_idx].item()
                confidence = scores[i][k_idx].item()

                if k_idx == 0:  # Top-1 match
                    top_1_match = {
                        "image": retrieval_dataset.get_original_image(subject_id=gallery_id),
                        "pred_label": gallery_labels[gallery_idx].item(),
                        "confidence": confidence
                    }

                if correct:
                    if most_confident_correct is None or confidence > most_confident_correct['confidence']:
                        most_confident_correct = {
                            "image": retrieval_dataset.get_original_image(subject_id=gallery_id),
                            "pred_label": gallery_labels[gallery_idx].item(),
                            "confidence": confidence,
                            "rank": k_idx + 1
                        }
                    if least_confident_correct is None or confidence < least_confident_correct['confidence']:
                        least_confident_correct = {
                            "image": retrieval_dataset.get_original_image(subject_id=gallery_id),
                            "pred_label": gallery_labels[gallery_idx].item(),
                            "confidence": confidence,
                            "rank": k_idx + 1
                        }

            samples.append({
                "query_image": retrieval_dataset.get_original_image(subject_id=query_id),
                "true_label": query_labels[i].item(),
                "top_1_match": top_1_match,
                "most_confident_correct": most_confident_correct,
                "least_confident_correct": least_confident_correct
            })

        # Step 4: Visualize results
        fig, axes = plt.subplots(n_vis, 4, figsize=(16, 3 * n_vis))

        column_titles = ["Query", "Top-1", "Most Confident Correct", "Least Confident Correct"]
        for i, title in enumerate(column_titles):
            axes[0, i].text(0.5, 1.2, title, fontsize=14, weight="bold", ha="center", transform=axes[0, i].transAxes)

        for i, sample in enumerate(samples):
            axes[i, 0].imshow(sample['query_image'])
            axes[i, 0].set_title(f"True Label: {sample['true_label']}", fontsize=10)
            axes[i, 0].axis("off")

            if sample['top_1_match']:
                axes[i, 1].imshow(sample['top_1_match']['image'])
                axes[i, 1].set_title(f"P: {sample['top_1_match']['pred_label']}\nConf: {sample['top_1_match']['confidence']:.2f}", fontsize=8)
            axes[i, 1].axis("off")

            if sample['most_confident_correct']:
                mcc = sample['most_confident_correct']
                axes[i, 2].imshow(mcc['image'])
                axes[i, 2].set_title(f"(k={mcc['rank']}) P: {mcc['pred_label']}\nConf: {mcc['confidence']:.2f}", fontsize=8)
            else:
                axes[i, 2].set_title("No Correct Match in top 600", fontsize=8)
            axes[i, 2].axis("off")

            if sample['least_confident_correct']:
                lcc = sample['least_confident_correct']
                axes[i, 3].imshow(lcc['image'])
                axes[i, 3].set_title(f"(k={lcc['rank']}) P: {lcc['pred_label']}\nConf: {lcc['confidence']:.2f}", fontsize=8)
            else:
                axes[i, 3].set_title("No Correct Match in top 600", fontsize=8)
            axes[i, 3].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
        plt.savefig(self.experiment_dir / 'visualize_matches.png')

    def make_tsne(self, embeddings, labels):
        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        tsne_result = tsne.fit_transform(embeddings.cpu())

        # Get labels for coloring and bin them
        y = labels.cpu().numpy()
        bins = pd.cut(pd.Series(y), bins=range(0, int(y.max()) + 25, 25), labels=[f'{i}-{i+24}' for i in range(0, int(y.max()), 25)])

        # Create a DataFrame for plotting
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': bins})

        # Plot the t-SNE result
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='tsne_1', y='tsne_2',
            hue='label',
            palette=sns.color_palette("hsv", len(tsne_result_df['label'].unique())),
            data=tsne_result_df,
            legend="full",
            alpha=0.7
        )
        plt.title('t-SNE of Gallery Embeddings')
        plt.show()
        plt.savefig(self.experiment_dir / 'tsne.png')

    def one_out_retrieval(self, model, loader, print = False, ba = None, ch = None, intervention_fn = None):
        embeddings, labels = model.collect_embeddings(loader, ba, ch, intervention_fn = intervention_fn)
        similarity_matrix = self.cosine_similarity_matrix(embeddings)
        recalls = {k: self.compute_recall_at_k(similarity_matrix, labels, labels, k=k) for k in [1, 5, 20, 100]}
        if print: print(f"One-out Recall@1: {recalls[1]*100:.2f}% - Recall@5: {recalls[5]*100:.2f}% - Recall@20: {recalls[20]*100:.2f}% - Recall@100: {recalls[100]*100:.2f}%")
        return recalls
