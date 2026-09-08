from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from seek_code import SEEK


@dataclass
class ElephantBookOutputs:
    backbone_embeddings: torch.Tensor
    seek_codes: torch.Tensor
    labels: torch.Tensor


class ElephantBookBaseline:
    """CurveRank-style linear fusion using backbone confidence and SEEK codes."""

    def __init__(
        self,
        backbone_weight: float = 0.5,
        wildcard_distance: float = 0.6,
        query_chunk_size: int = 512,
    ):
        if not 0.0 <= backbone_weight <= 1.0:
            raise ValueError(f"backbone_weight must be in [0, 1], got {backbone_weight}")
        if wildcard_distance < 0:
            raise ValueError(f"wildcard_distance must be non-negative, got {wildcard_distance}")
        if query_chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

        self.backbone_weight = backbone_weight
        self.seek_weight = 1.0 - backbone_weight
        self.wildcard_distance = wildcard_distance
        self.query_chunk_size = query_chunk_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        wildcard_ids = []
        for attr_name in SEEK.attribute_names:
            mapping = SEEK.mappings[attr_name]
            wildcard_ids.append(mapping.index("_") if "_" in mapping else -1)
        self.wildcard_ids = torch.tensor(wildcard_ids, dtype=torch.long)

    def collect_outputs(
        self,
        loader,
        backbone,
        backbone_for_concepts,
        concept_head,
        correction_fn=None,
    ) -> ElephantBookOutputs:
        backbone.freeze()
        backbone_for_concepts.freeze()
        concept_head.freeze()

        backbone_embeddings = []
        seek_codes = []
        labels = []

        for batch in tqdm(loader, desc="Collecting ElephantBook inputs"):
            images = batch[0].to(self.device)
            ele_id_label = batch[2].to(self.device)
            subject_seek = batch[4]
            sighting_seek = batch[5]
            left_ears = batch[6].to(self.device)
            right_ears = batch[7].to(self.device)
            idx = batch[10]

            with torch.no_grad():
                visual_embeddings = backbone.forward(images, left_ears, right_ears)

                if correction_fn is SEEK.perfect_correction or correction_fn is SEEK.oracle_correction:
                    predicted_seek = None
                else:
                    concept_embeddings = backbone_for_concepts.forward(images, left_ears, right_ears)
                    concept_logits = concept_head.layer(concept_embeddings)
                    predicted_seek = SEEK.closest_valid_one_hot(concept_logits)

                if correction_fn is not None:
                    predicted_seek = correction_fn(
                        predicted_seek,
                        subject_seek,
                        sighting_seek,
                        ele_id=ele_id_label,
                        idx=idx,
                    )

                backbone_embeddings.append(visual_embeddings.detach().cpu())
                seek_codes.append(predicted_seek.detach().cpu())
                labels.append(ele_id_label.detach().cpu())

        return ElephantBookOutputs(
            backbone_embeddings=torch.cat(backbone_embeddings, dim=0),
            seek_codes=torch.cat(seek_codes, dim=0),
            labels=torch.cat(labels, dim=0).long(),
        )

    def one_hot_to_categories(self, seek_codes: torch.Tensor) -> torch.Tensor:
        categories = []
        offset = 0
        for attr_name in SEEK.attribute_names:
            length = SEEK.lengths[attr_name]
            categories.append(torch.argmax(seek_codes[:, offset : offset + length], dim=1))
            offset += length
        return torch.stack(categories, dim=1)

    def seek_distance_matrix(
        self,
        query_seek_codes: torch.Tensor,
        gallery_seek_codes: torch.Tensor,
    ) -> torch.Tensor:
        query_categories = self.one_hot_to_categories(query_seek_codes)
        gallery_categories = self.one_hot_to_categories(gallery_seek_codes)
        wildcard_ids = self.wildcard_ids.to(query_categories.device)

        distance = torch.zeros(
            query_categories.size(0),
            gallery_categories.size(0),
            device=query_categories.device,
            dtype=torch.float32,
        )

        for attr_idx in range(len(SEEK.attribute_names)):
            query_attr = query_categories[:, attr_idx].unsqueeze(1)
            gallery_attr = gallery_categories[:, attr_idx].unsqueeze(0)
            wildcard_id = wildcard_ids[attr_idx]

            differs = query_attr != gallery_attr
            if wildcard_id >= 0:
                has_wildcard = (query_attr == wildcard_id) | (gallery_attr == wildcard_id)
            else:
                has_wildcard = torch.zeros_like(differs)

            attr_distance = torch.where(
                has_wildcard,
                torch.full_like(distance, self.wildcard_distance),
                differs.to(torch.float32),
            )
            distance = distance + attr_distance

        return distance

    def score_matrices(
        self,
        query_backbone: torch.Tensor,
        gallery_backbone: torch.Tensor,
        query_seek: torch.Tensor,
        gallery_seek: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_backbone = F.normalize(query_backbone, dim=1)
        gallery_backbone = F.normalize(gallery_backbone, dim=1)
        backbone_confidence = (query_backbone @ gallery_backbone.T + 1.0) / 2.0

        seek_distance = self.seek_distance_matrix(query_seek, gallery_seek)
        seek_similarity = 1.0 - (seek_distance / len(SEEK.attribute_names))
        combined = self.backbone_weight * backbone_confidence + self.seek_weight * seek_similarity

        return {
            "combined": combined,
            "backbone": backbone_confidence,
            "seek": seek_similarity,
            "seek_distance": seek_distance,
        }

    @staticmethod
    def recall_counts(score_matrix, query_labels, gallery_labels, ks):
        counts = {k: 0 for k in ks}
        for k in ks:
            effective_k = min(k, score_matrix.size(1))
            topk_indices = torch.topk(score_matrix, k=effective_k, dim=1).indices
            topk_labels = gallery_labels[topk_indices]
            counts[k] += (topk_labels == query_labels.unsqueeze(1)).any(dim=1).sum().item()
        return counts

    def evaluate_scores(self, query_outputs: ElephantBookOutputs, gallery_outputs: ElephantBookOutputs):
        ks = [1, 5, 10, 20, 100]
        totals = {
            "combined": {k: 0 for k in ks},
            "backbone": {k: 0 for k in ks},
            "seek": {k: 0 for k in ks},
        }
        distance_sum = 0.0
        distance_pairs = 0

        gallery_backbone = gallery_outputs.backbone_embeddings.to(self.device)
        gallery_seek = gallery_outputs.seek_codes.to(self.device)
        gallery_labels = gallery_outputs.labels.to(self.device)

        for start in range(0, query_outputs.labels.size(0), self.query_chunk_size):
            end = start + self.query_chunk_size
            query_backbone = query_outputs.backbone_embeddings[start:end].to(self.device)
            query_seek = query_outputs.seek_codes[start:end].to(self.device)
            query_labels = query_outputs.labels[start:end].to(self.device)

            matrices = self.score_matrices(
                query_backbone=query_backbone,
                gallery_backbone=gallery_backbone,
                query_seek=query_seek,
                gallery_seek=gallery_seek,
            )

            for name in ["combined", "backbone", "seek"]:
                counts = self.recall_counts(matrices[name], query_labels, gallery_labels, ks)
                for k in ks:
                    totals[name][k] += counts[k]

            distance_sum += matrices["seek_distance"].sum().item()
            distance_pairs += matrices["seek_distance"].numel()

        n_queries = query_outputs.labels.size(0)
        recalls = {
            name: {k: totals[name][k] / n_queries for k in ks}
            for name in ["combined", "backbone", "seek"]
        }
        average_seek_distance = distance_sum / distance_pairs if distance_pairs else 0.0
        return recalls, average_seek_distance

    def evaluate(
        self,
        backbone,
        backbone_for_concepts,
        concept_head,
        train_loader,
        test_loader,
        gallery_correction_fn=None,
        query_correction_fn=None,
    ):
        gallery_outputs = self.collect_outputs(
            train_loader,
            backbone=backbone,
            backbone_for_concepts=backbone_for_concepts,
            concept_head=concept_head,
            correction_fn=gallery_correction_fn,
        )
        query_outputs = self.collect_outputs(
            test_loader,
            backbone=backbone,
            backbone_for_concepts=backbone_for_concepts,
            concept_head=concept_head,
            correction_fn=query_correction_fn,
        )

        return self.evaluate_scores(query_outputs=query_outputs, gallery_outputs=gallery_outputs)
