from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .config import (
        BACKBONE_FOR_CONCEPTS_PRETRAINING,
        BACKBONE_PRETRAINING,
        CONCEPT_HEAD_PRETRAINING,
        DEFAULT_EXPERIMENT_DIR,
        METHODS_DIR,
    )
    from .seek_utils import one_hot_to_seek, seek_to_one_hot
except ImportError:
    from config import (
        BACKBONE_FOR_CONCEPTS_PRETRAINING,
        BACKBONE_PRETRAINING,
        CONCEPT_HEAD_PRETRAINING,
        DEFAULT_EXPERIMENT_DIR,
        METHODS_DIR,
    )
    from seek_utils import one_hot_to_seek, seek_to_one_hot

if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

os.environ.setdefault("WANDB_MODE", "disabled")

torch = None
F = None
Image = None
transforms = None
wandb = None
Backbone = None
ConceptHeadTunneled = None
ThreeHeadNN = None
categorical_CE_loss = None
EleHandler = None
Projector = None
SEEK = None


@dataclass
class GalleryCache:
    idxs: list[int]
    embeddings: object
    metadata: list[dict]


@dataclass
class ImageFeatureCache:
    predicted_seek: str = ""
    backbone_embedding: object = None
    projected_embedding: object = None
    projected_seek_code: str = ""
    projected_embeddings_by_seek: dict[str, object] = field(default_factory=dict)


class ModelService:
    def __init__(self, catalog, state, experiment_dir: Path = DEFAULT_EXPERIMENT_DIR):
        self.catalog = catalog
        self.state = state
        self.experiment_dir = Path(experiment_dir)
        self.device = None
        self.dataset = None
        self.backbone_for_concepts = None
        self.backbone = None
        self.concept_head = None
        self.projector = None
        self.gallery_cache = None
        self.upload_transform = None
        self.image_feature_cache = {}
        self.seek_projection_cache = {}

    def _ensure_model_imports(self):
        global torch, F, Image, transforms, wandb
        global Backbone, ConceptHeadTunneled, ThreeHeadNN, categorical_CE_loss
        global EleHandler, Projector, SEEK
        if torch is not None:
            return
        import torch as torch_module
        import torch.nn.functional as functional_module
        from PIL import Image as image_module
        import torchvision.transforms as transforms_module
        import wandb as wandb_module
        from backbone import Backbone as BackboneClass
        from concept_head_tunneled import (
            ConceptHeadTunneled as ConceptHeadTunneledClass,
            ThreeHeadNN as ThreeHeadNNClass,
            categorical_CE_loss as categorical_loss,
        )
        from data_handler import EleHandler as EleHandlerClass
        from projector import Projector as ProjectorClass
        from seek_code import SEEK as SEEKClass

        torch = torch_module
        F = functional_module
        Image = image_module
        transforms = transforms_module
        wandb = wandb_module
        Backbone = BackboneClass
        ConceptHeadTunneled = ConceptHeadTunneledClass
        ThreeHeadNN = ThreeHeadNNClass
        categorical_CE_loss = categorical_loss
        EleHandler = EleHandlerClass
        Projector = ProjectorClass
        SEEK = SEEKClass

    def load(self):
        if self.projector is not None:
            return
        self._ensure_model_imports()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upload_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        if wandb.run is None:
            wandb.init(mode="disabled")

        self.dataset = EleHandler(subset="2+encounters", dataset_type="mara")
        num_classes = len(self.dataset.ele_id_to_label)

        self.backbone_for_concepts = Backbone(
            model_name="MegaDescriptor",
            pretraining=BACKBONE_FOR_CONCEPTS_PRETRAINING,
            num_classes=num_classes,
            experiment_code="ui",
        )
        self.backbone = Backbone(
            model_name="MegaDescriptor",
            pretraining=BACKBONE_PRETRAINING,
            num_classes=num_classes,
            experiment_code="ui",
        )
        self.concept_head = ConceptHeadTunneled(
            loss=categorical_CE_loss,
            experiment_code="ui",
            layer=ThreeHeadNN(),
            reset_weights=False,
            pretraining=CONCEPT_HEAD_PRETRAINING,
        )
        self.projector = Projector(
            loss_type="ArcFace",
            experiment_code="ui",
            reset_weights=True,
            alpha=0.5,
            alpha_learnable=True,
            layer_norm=True,
            network_type="small",
            num_classes=num_classes,
        )
        self._load_projector_checkpoint()

        self.backbone_for_concepts.freeze()
        self.backbone.freeze()
        self.concept_head.freeze()
        self.projector.freeze()

    def _load_projector_checkpoint(self):
        projector_path = self.experiment_dir / "projector.pt"
        backbone_path = self.experiment_dir / "backbone_w.pt"
        alpha_path = self.experiment_dir / "alpha.pt"
        if not projector_path.exists():
            raise FileNotFoundError(f"Missing projector checkpoint: {projector_path}")
        if not backbone_path.exists():
            raise FileNotFoundError(f"Missing projector backbone checkpoint: {backbone_path}")

        self.projector.layer.load_state_dict(torch.load(projector_path, map_location=self.device))
        self.backbone.layer.load_state_dict(torch.load(backbone_path, map_location=self.device), strict=False)
        if alpha_path.exists():
            alpha_data = torch.load(alpha_path, map_location=self.device)
            alpha = float(alpha_data["alpha"] if isinstance(alpha_data, dict) else alpha_data)
            if hasattr(self.projector.alpha, "data"):
                self.projector.alpha.data.fill_(alpha)
            else:
                self.projector.alpha = alpha

    def _alpha(self):
        alpha = self.projector.alpha
        if torch.is_tensor(alpha):
            return alpha.to(self.device)
        return torch.tensor(float(alpha), device=self.device)

    def _dataset_item(self, idx):
        item = self.dataset[int(idx)]
        return {
            "idx": int(idx),
            "image": item[0],
            "subject_id": str(item[1]),
            "ele_label": item[2],
            "subject_seek_1h": item[4],
            "ele_seek_1h": item[5],
            "left_ear": item[6],
            "right_ear": item[7],
            "subject_seek": item[8],
            "ele_seek": item[9],
        }

    def _stack_dataset_items(self, idxs):
        items = [self._dataset_item(idx) for idx in idxs]
        return {
            "items": items,
            "images": torch.stack([item["image"] for item in items]).to(self.device),
            "left_ears": torch.stack([item["left_ear"] for item in items]).to(self.device),
            "right_ears": torch.stack([item["right_ear"] for item in items]).to(self.device),
        }

    def _feature_entry(self, image_key):
        image_key = str(image_key)
        if image_key not in self.image_feature_cache:
            self.image_feature_cache[image_key] = ImageFeatureCache()
        return self.image_feature_cache[image_key]

    def _dataset_image_key(self, idx):
        return f"dataset:{int(idx)}"

    def _upload_image_key(self, upload_row):
        return str(upload_row["query_id"])

    def remember_query_prediction(self, query_id, predicted_seek):
        if not predicted_seek:
            return
        self._feature_entry(str(query_id)).predicted_seek = str(predicted_seek)

    def _store_backbone_embeddings(self, image_keys, images, left_ears, right_ears):
        missing = [
            pos for pos, image_key in enumerate(image_keys)
            if self._feature_entry(image_key).backbone_embedding is None
        ]
        if not missing:
            return
        with torch.no_grad():
            embeddings = self.backbone.forward(
                images[missing], left_ears[missing], right_ears[missing]
            )
            if self.projector.layer_norm:
                embeddings = F.layer_norm(embeddings, embeddings.size()[1:])
        for offset, pos in enumerate(missing):
            self._feature_entry(image_keys[pos]).backbone_embedding = embeddings[offset].detach().cpu().unsqueeze(0)

    def _store_concept_predictions(self, image_keys, images, left_ears, right_ears):
        missing = [
            pos for pos, image_key in enumerate(image_keys)
            if not self._feature_entry(image_key).predicted_seek
        ]
        if not missing:
            return
        with torch.no_grad():
            embeddings = self.backbone_for_concepts.forward(
                images[missing], left_ears[missing], right_ears[missing]
            )
            logits = self.concept_head.layer(embeddings)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            predicted = SEEK.closest_valid_one_hot(logits)
        for offset, pos in enumerate(missing):
            one_hot = predicted[offset].detach().cpu()
            self._feature_entry(image_keys[pos]).predicted_seek = one_hot_to_seek(one_hot)

    def _seek_projection(self, seek_code):
        seek_code = str(seek_code)
        if seek_code in self.seek_projection_cache:
            return self.seek_projection_cache[seek_code]
        seek_tensor = seek_to_one_hot(seek_code).unsqueeze(0).to(self.device)
        with torch.no_grad():
            projected = self.projector.layer(seek_tensor)
            if self.projector.layer_norm:
                projected = F.layer_norm(projected, projected.size()[1:])
        projected = projected.detach().cpu()
        self.seek_projection_cache[seek_code] = projected
        return projected

    def _alpha_float(self):
        return float(self._alpha().detach().cpu())

    def _cached_projected_embedding(self, image_key, seek_code):
        entry = self._feature_entry(image_key)
        seek_code = str(seek_code)
        if seek_code in entry.projected_embeddings_by_seek:
            embedding = entry.projected_embeddings_by_seek[seek_code]
        else:
            if entry.backbone_embedding is None:
                raise RuntimeError(f"Missing cached backbone embedding for {image_key}")
            projected_concepts = self._seek_projection(seek_code)
            alpha = self._alpha_float()
            embedding = (1 - alpha) * entry.backbone_embedding + alpha * projected_concepts
            entry.projected_embeddings_by_seek[seek_code] = embedding
        entry.projected_seek_code = seek_code
        entry.projected_embedding = embedding
        return embedding

    def _dataset_projected_embeddings(self, idxs, seek_codes):
        self.load()
        idxs = [int(idx) for idx in idxs]
        image_keys = [self._dataset_image_key(idx) for idx in idxs]
        missing = [
            idx for idx, image_key in zip(idxs, image_keys)
            if self._feature_entry(image_key).backbone_embedding is None
        ]
        if missing:
            batch = self._stack_dataset_items(missing)
            self._store_backbone_embeddings(
                [self._dataset_image_key(idx) for idx in missing],
                batch["images"],
                batch["left_ears"],
                batch["right_ears"],
            )
        return torch.cat([
            self._cached_projected_embedding(image_key, seek_code)
            for image_key, seek_code in zip(image_keys, seek_codes)
        ], dim=0)

    def _upload_projected_embedding(self, upload_row, seek_code):
        self.load()
        image_key = self._upload_image_key(upload_row)
        entry = self._feature_entry(image_key)
        if entry.backbone_embedding is None:
            images, left_ears, right_ears = self._upload_tensors(
                upload_row["body_image_path"],
                upload_row.get("left_ear_path", ""),
                upload_row.get("right_ear_path", ""),
            )
            self._store_backbone_embeddings([image_key], images, left_ears, right_ears)
        return self._cached_projected_embedding(image_key, seek_code)

    def warm_dataset_query_features(self, idx, seek_code=None):
        predicted = self.predict_seek_for_dataset_idx(idx)
        self._dataset_projected_embeddings([idx], [seek_code or predicted])
        return predicted

    def warm_upload_query_features(self, upload_row, seek_code=None):
        predicted = self.predict_seek_for_upload(upload_row)
        self._upload_projected_embedding(upload_row, seek_code or predicted)
        return predicted

    def _embed_with_seek(self, images, left_ears, right_ears, seek_vectors):
        seek_vectors = seek_vectors.to(self.device)
        with torch.no_grad():
            image_embeddings = self.backbone.forward(images, left_ears, right_ears)
            projected_concepts = self.projector.layer(seek_vectors)
            if self.projector.layer_norm:
                image_embeddings = F.layer_norm(image_embeddings, image_embeddings.size()[1:])
                projected_concepts = F.layer_norm(projected_concepts, projected_concepts.size()[1:])
            edited = (1 - self._alpha()) * image_embeddings + self._alpha() * projected_concepts
        return edited

    def predict_seek_for_dataset_idx(self, idx):
        self.load()
        image_key = self._dataset_image_key(idx)
        entry = self._feature_entry(image_key)
        if entry.predicted_seek:
            return entry.predicted_seek
        batch = self._stack_dataset_items([idx])
        self._store_concept_predictions(
            [image_key], batch["images"], batch["left_ears"], batch["right_ears"]
        )
        return entry.predicted_seek

    def predict_seek_for_dataset_idxs(self, idxs, batch_size=32):
        self.load()
        results = {}
        idxs = [int(idx) for idx in idxs]
        missing = []
        for idx in idxs:
            image_key = self._dataset_image_key(idx)
            predicted = self._feature_entry(image_key).predicted_seek
            if predicted:
                results[int(idx)] = predicted
            else:
                missing.append(idx)
        for start in range(0, len(missing), batch_size):
            batch_idxs = missing[start:start + batch_size]
            batch = self._stack_dataset_items(batch_idxs)
            image_keys = [self._dataset_image_key(idx) for idx in batch_idxs]
            self._store_concept_predictions(
                image_keys, batch["images"], batch["left_ears"], batch["right_ears"]
            )
            for idx, image_key in zip(batch_idxs, image_keys):
                results[int(idx)] = self._feature_entry(image_key).predicted_seek
        return results

    def _upload_tensors(self, body_path, left_ear_path="", right_ear_path=""):
        image = Image.open(body_path).convert("RGB")
        image_tensor = self.upload_transform(image)
        zero_ear = torch.zeros_like(image_tensor)
        left = self.upload_transform(Image.open(left_ear_path).convert("RGB")) if left_ear_path else zero_ear
        right = self.upload_transform(Image.open(right_ear_path).convert("RGB")) if right_ear_path else zero_ear
        return (
            image_tensor.unsqueeze(0).to(self.device),
            left.unsqueeze(0).to(self.device),
            right.unsqueeze(0).to(self.device),
        )

    def predict_seek_for_upload(self, upload_row):
        self.load()
        image_key = self._upload_image_key(upload_row)
        entry = self._feature_entry(image_key)
        if entry.predicted_seek:
            return entry.predicted_seek
        images, left_ears, right_ears = self._upload_tensors(
            upload_row["body_image_path"],
            upload_row.get("left_ear_path", ""),
            upload_row.get("right_ear_path", ""),
        )
        self._store_concept_predictions([image_key], images, left_ears, right_ears)
        return entry.predicted_seek

    def refresh_gallery_cache(self, force=False, batch_size=32):
        self.load()
        if self.gallery_cache is not None and not force:
            return self.gallery_cache

        gallery = self.catalog.gallery_dataframe(self.state)
        idxs = gallery["idx"].astype(int).tolist()
        embeddings = []
        metadata = []

        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start:start + batch_size]
            batch = self._stack_dataset_items(batch_idxs)
            image_keys = [self._dataset_image_key(idx) for idx in batch_idxs]
            self._store_backbone_embeddings(
                image_keys, batch["images"], batch["left_ears"], batch["right_ears"]
            )
            self._store_concept_predictions(
                image_keys, batch["images"], batch["left_ears"], batch["right_ears"]
            )
            seek_codes = []
            for item in batch["items"]:
                row = gallery[gallery["idx"] == item["idx"]].iloc[0]
                seek_codes.append(row["current_ele_seek"])
                metadata.append({
                    "idx": item["idx"],
                    "subject_id": item["subject_id"],
                    "ele_id": str(self.catalog.df.iloc[item["idx"]]["ele_id"]),
                    "encounter_id": str(self.catalog.df.iloc[item["idx"]]["encounter_id"]),
                    "subject_seek": item["subject_seek"],
                    "current_ele_seek": row["current_ele_seek"],
                })
            embeddings.append(self._dataset_projected_embeddings(batch_idxs, seek_codes).detach().cpu())

        if embeddings:
            gallery_embeddings = torch.cat(embeddings, dim=0)
        else:
            gallery_embeddings = torch.empty(0, 2304)

        self.gallery_cache = GalleryCache(idxs=idxs, embeddings=gallery_embeddings, metadata=metadata)
        return self.gallery_cache

    def update_gallery_cache_for_idxs(self, idxs, batch_size=32):
        if self.gallery_cache is None:
            return None
        self.load()
        idxs = [int(idx) for idx in idxs if int(idx) in self.gallery_cache.idxs]
        if not idxs:
            return self.gallery_cache

        gallery = self.catalog.gallery_dataframe(self.state)
        for start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[start:start + batch_size]
            batch = self._stack_dataset_items(batch_idxs)
            image_keys = [self._dataset_image_key(idx) for idx in batch_idxs]
            self._store_backbone_embeddings(
                image_keys, batch["images"], batch["left_ears"], batch["right_ears"]
            )
            self._store_concept_predictions(
                image_keys, batch["images"], batch["left_ears"], batch["right_ears"]
            )
            seek_codes = []
            rows = []
            for item in batch["items"]:
                row = gallery[gallery["idx"] == item["idx"]]
                if row.empty:
                    seek_codes.append(item["ele_seek"])
                    rows.append(None)
                    continue
                row = row.iloc[0]
                seek_codes.append(row["current_ele_seek"])
                rows.append(row)
            embeddings = self._dataset_projected_embeddings(batch_idxs, seek_codes).detach().cpu()
            for i, item in enumerate(batch["items"]):
                cache_pos = self.gallery_cache.idxs.index(item["idx"])
                self.gallery_cache.embeddings[cache_pos] = embeddings[i]
                if rows[i] is not None:
                    self.gallery_cache.metadata[cache_pos]["current_ele_seek"] = rows[i]["current_ele_seek"]
        return self.gallery_cache

    def rank_dataset_query(self, idx, seek_code, top_k=10):
        self.load()
        cache = self.refresh_gallery_cache()
        query_embedding = self._dataset_projected_embeddings([idx], [seek_code]).detach().cpu()
        return self._rank_embedding(query_embedding, cache, top_k)

    def rank_upload_query(self, upload_row, seek_code, top_k=10):
        self.load()
        cache = self.refresh_gallery_cache()
        query_embedding = self._upload_projected_embedding(upload_row, seek_code).detach().cpu()
        return self._rank_embedding(query_embedding, cache, top_k)

    def _rank_embedding(self, query_embedding, cache, top_k):
        if cache.embeddings.numel() == 0:
            return []
        q = F.normalize(query_embedding, dim=1)
        g = F.normalize(cache.embeddings, dim=1)
        scores = torch.matmul(q, g.T).squeeze(0)
        k = min(int(top_k), scores.numel())
        values, indices = torch.topk(scores, k=k)
        results = []
        for rank, (score, gallery_pos) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            meta = dict(cache.metadata[gallery_pos])
            meta["rank"] = rank
            meta["score"] = float(score)
            results.append(meta)
        return results
