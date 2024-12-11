import torch
from tqdm import tqdm
from seek_code import SEEK
import torch.nn.functional as F

class Retrieval:
    def __init__(self):
        pass

    def collect_edited_embeddings(loader, backbone, concept_head, projector, intervention_fn =None):
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

    def collect_MD_embeddings(loader, backbone, intervention_fn =None):

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

    def cosine_similarity_matrix(query_embeddings, gallery_embeddings):
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

        return similarity_matrix


    def compute_retrieval_accuracy(similarity_matrix, query_labels, gallery_labels, k_values=(1, 5)):
        """
        Compute top-k retrieval accuracy using a precomputed similarity matrix.

        Args:
            similarity_matrix (torch.Tensor): Precomputed similarity matrix (num_queries, num_gallery).
            query_labels (torch.Tensor): Tensor of query labels (num_queries,).
            gallery_labels (torch.Tensor): Tensor of gallery labels (num_gallery,).
            k_values (tuple): Top-k values to compute accuracy for (default: top-1 and top-5).

        Returns:
            dict: Dictionary with top-k accuracy values.
        """
        # Sort gallery indices by similarity (descending order)
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)  # Shape: (num_queries, num_gallery)

        # Initialize accuracy dictionary
        topk_accuracies = {f"top-{k}": 0 for k in k_values}

        # Compute top-k accuracies
        for k in k_values:
            # Get top-k indices for each query
            topk_indices = sorted_indices[:, :k]  # Shape: (num_queries, k)
            
            # Check if ground truth label is in the top-k predictions
            correct_matches = 0
            for i, query_label in enumerate(query_labels):
                if query_label in gallery_labels[topk_indices[i]]:
                    correct_matches += 1
            
            # Compute accuracy
            topk_accuracies[f"top-{k}"] = correct_matches / len(query_labels)

        return topk_accuracies        



    def perform_retrieval(gallery_loader, query_loader, ba, ch = None, pr = None, k_values=(1, 5, 20, 100)):
        
        if ch is None and pr is None:
            gallery_embeddings, gallery_labels = Retrieval.collect_MD_embeddings(gallery_loader, ba)
            query_embeddings, query_labels = Retrieval.collect_MD_embeddings(query_loader, ba)
        else:
            gallery_embeddings, gallery_labels = Retrieval.collect_edited_embeddings(gallery_loader, ba, ch, pr)
            query_embeddings, query_labels =  Retrieval.collect_edited_embeddings(query_loader, ba, ch, pr)

        #gallery_embeddings, gallery_labels = collect_MD_embeddings(gallery_loader, ba)
        #query_embeddings, query_labels = collect_MD_embeddings(query_loader, ba)

        # Step 1: Compute the similarity matrix
        similarity_matrix =  Retrieval.cosine_similarity_matrix(query_embeddings, gallery_embeddings)

        #print(similarity_matrix)

        # Step 2: Compute top-k retrieval accuracy
        accuracies =  Retrieval.compute_retrieval_accuracy(similarity_matrix, query_labels, gallery_labels, k_values=k_values)
        print(accuracies)