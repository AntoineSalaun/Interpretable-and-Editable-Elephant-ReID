import torch
from tqdm import tqdm
from seek_code import SEEK
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Retrieval:
    def __init__(self):
        pass

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


    from torch.nn import functional as F

    def cosine_similarity_matrix(self, query_embeddings, gallery_embeddings):
        """
        Compute the cosine similarity matrix between query and gallery embeddings.

        Args:
            query_embeddings (torch.Tensor): Tensor of query embeddings (num_queries, embedding_dim).
            gallery_embeddings (torch.Tensor): Tensor of gallery embeddings (num_gallery, embedding_dim).

        Returns:
            torch.Tensor: Similarity matrix (num_queries, num_gallery).
        """
        # Normalize embeddings for cosine similarity
        query_embeddings = F.normalize(query_embeddings, dim=1)
        gallery_embeddings = F.normalize(gallery_embeddings, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(query_embeddings, gallery_embeddings.T)

        print('gallery embeddings shape:', gallery_embeddings.shape)
        print('query embeddings shape:', query_embeddings.shape)
        print('similarity matrix shape:', similarity_matrix.shape)

        #plt.hist(similarity_matrix.flatten().cpu().numpy(), bins=100)   
        #plt.show()

        return similarity_matrix


    def compute_retrieval_accuracy(self, similarity_matrix, query_labels, gallery_labels, top_k=1):
        """
        Compute retrieval accuracy from a cosine similarity matrix.

        Args:
            similarity_matrix (torch.Tensor): Similarity matrix (num_queries, num_gallery).
            query_labels (torch.Tensor): Labels for query embeddings (num_queries,).
            gallery_labels (torch.Tensor): Labels for gallery embeddings (num_gallery,).
            top_k (int): Number of top matches to consider for accuracy.

        Returns:
            float: Retrieval accuracy as a percentage.
        """
        # Get the indices of the top-k gallery items for each query
        top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1, largest=True).indices
        
        # Match query labels with gallery labels for the top-k indices
        correct_matches = 0
        for i, query_label in enumerate(query_labels):
            top_k_labels = gallery_labels[top_k_indices[i]]
            if query_label in top_k_labels:
                correct_matches += 1
        
        # Calculate accuracy
        accuracy = correct_matches / len(query_labels)
        return accuracy
        
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


    def perform_retrieval(self, query_loader, gallery_loader, backbone, concept_head= None, projector= None, k_list = [1, 5, 10]):
        
        if concept_head is not None and projector is not None:
            query_embeddings, query_labels = self.collect_edited_embeddings(query_loader, backbone, concept_head, projector)
            gallery_embeddings, gallery_labels = self.collect_edited_embeddings(gallery_loader, backbone, concept_head, projector)
        else:
            query_embeddings, query_labels = self.collect_MD_embeddings(query_loader, backbone)
            gallery_embeddings, gallery_labels = self.collect_MD_embeddings(gallery_loader, backbone)
        
        similarity_matrix = self.cosine_similarity_matrix(query_embeddings, gallery_embeddings)
        
        results = {}

        for k in k_list:
            # Use the recall and accuracy functions defined earlier
            recall = self.compute_recall_at_k(similarity_matrix, query_labels, gallery_labels, k=k)
            accuracy = self.compute_retrieval_accuracy(similarity_matrix, query_labels, gallery_labels, top_k=k)

            # Store the results
            results[k] = {"recall": recall, "accuracy": accuracy}

        print(results)
        return results
