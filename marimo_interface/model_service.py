from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    version: str
    idxs: list[int]
    embeddings: object
    metadata: list[dict]


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
        self.upload_transform = None
        self.gallery_cache = None

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

    def load(self) -> None:
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
            experiment_code="marimo_ui",
        )
        self.backbone = Backbone(
            model_name="MegaDescriptor",
            pretraining=BACKBONE_PRETRAINING,
            num_classes=num_classes,
            experiment_code="marimo_ui",
        )
        self.concept_head = ConceptHeadTunneled(
            loss=categorical_CE_loss,
            experiment_code="marimo_ui",
            layer=ThreeHeadNN(),
            reset_weights=False,
            pretraining=CONCEPT_HEAD_PRETRAINING,
        )
        self.projector = Projector(
            loss_type="ArcFace",
            experiment_code="marimo_ui",
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

    def _load_projector_checkpoint(self) -> None:
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

    def _alpha_float(self) -> float:
        return float(self._alpha().detach().cpu())

    def _dataset_image_key(self, idx) -> str:
        return f"dataset:{int(idx)}"

    def _upload_image_key(self, upload_row) -> str:
        return str(upload_row["query_id"])

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

    def _feature(self, image_key):
        return self.state.get_feature(image_key) or {}

    def remember_query_prediction(self, query_id, predicted_seek) -> None:
        if predicted_seek:
            image_type = "dataset" if str(query_id).startswith("dataset:") else "upload"
            self.state.upsert_feature(query_id, image_type, source_id=str(query_id), predicted_seek=str(predicted_seek))

    def _store_backbone_embeddings(self, image_keys, images, left_ears, right_ears, image_type="dataset"):
        missing_positions = [
            pos for pos, image_key in enumerate(image_keys)
            if self._feature(image_key).get("backbone_embedding") is None
        ]
        if not missing_positions:
            return
        with torch.no_grad():
            embeddings = self.backbone.forward(
                images[missing_positions],
                left_ears[missing_positions],
                right_ears[missing_positions],
            )
            if self.projector.layer_norm:
                embeddings = F.layer_norm(embeddings, embeddings.size()[1:])
        for offset, pos in enumerate(missing_positions):
            image_key = image_keys[pos]
            predicted = self._feature(image_key).get("predicted_seek", "")
            self.state.upsert_feature(
                image_key,
                image_type=image_type,
                source_id=image_key,
                predicted_seek=predicted,
                backbone_embedding=embeddings[offset].detach().cpu().numpy()[None, :],
            )

    def _store_concept_predictions(self, image_keys, images, left_ears, right_ears, image_type="dataset"):
        missing_positions = [
            pos for pos, image_key in enumerate(image_keys)
            if not self._feature(image_key).get("predicted_seek")
        ]
        if not missing_positions:
            return
        with torch.no_grad():
            embeddings = self.backbone_for_concepts.forward(
                images[missing_positions],
                left_ears[missing_positions],
                right_ears[missing_positions],
            )
            logits = self.concept_head.layer(embeddings)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            predicted = SEEK.closest_valid_one_hot(logits)
        for offset, pos in enumerate(missing_positions):
            image_key = image_keys[pos]
            existing_embedding = self._feature(image_key).get("backbone_embedding")
            self.state.upsert_feature(
                image_key,
                image_type=image_type,
                source_id=image_key,
                predicted_seek=one_hot_to_seek(predicted[offset].detach().cpu()),
                backbone_embedding=existing_embedding,
            )

    def _seek_projection(self, seek_code):
        seek_code = str(seek_code)
        cached = self.state.get_seek_projection(seek_code)
        if cached is not None:
            return torch.as_tensor(cached, dtype=torch.float32)
        seek_tensor = seek_to_one_hot(seek_code).unsqueeze(0).to(self.device)
        with torch.no_grad():
            projected = self.projector.layer(seek_tensor)
            if self.projector.layer_norm:
                projected = F.layer_norm(projected, projected.size()[1:])
        projected_np = projected.detach().cpu().numpy()
        self.state.put_seek_projection(seek_code, projected_np)
        return torch.as_tensor(projected_np, dtype=torch.float32)

    def _projected_embedding(self, image_key, seek_code):
        cached = self.state.get_projected_embedding(image_key, seek_code)
        if cached is not None:
            return torch.as_tensor(cached, dtype=torch.float32)
        feature = self._feature(image_key)
        backbone = feature.get("backbone_embedding")
        if backbone is None:
            raise RuntimeError(f"Missing cached backbone embedding for {image_key}")
        projected_concepts = self._seek_projection(seek_code).numpy()
        embedding = (1 - self._alpha_float()) * np.asarray(backbone) + self._alpha_float() * projected_concepts
        self.state.put_projected_embedding(image_key, seek_code, embedding)
        return torch.as_tensor(embedding, dtype=torch.float32)

    def _dataset_projected_embeddings(self, idxs, seek_codes, batch_size=64):
        self.load()
        idxs = [int(idx) for idx in idxs]
        image_keys = [self._dataset_image_key(idx) for idx in idxs]
        missing = [
            idx for idx, image_key in zip(idxs, image_keys)
            if self._feature(image_key).get("backbone_embedding") is None
        ]
        for start in range(0, len(missing), batch_size):
            batch_idxs = missing[start:start + batch_size]
            batch = self._stack_dataset_items(batch_idxs)
            self._store_backbone_embeddings(
                [self._dataset_image_key(idx) for idx in batch_idxs],
                batch["images"],
                batch["left_ears"],
                batch["right_ears"],
                image_type="dataset",
            )
        return torch.cat([
            self._projected_embedding(image_key, seek_code)
            for image_key, seek_code in zip(image_keys, seek_codes)
        ], dim=0)

    def _upload_projected_embedding(self, upload_row, seek_code):
        self.load()
        image_key = self._upload_image_key(upload_row)
        if self._feature(image_key).get("backbone_embedding") is None:
            images, left_ears, right_ears = self._upload_tensors(
                upload_row["body_image_path"],
                upload_row.get("left_ear_path", ""),
                upload_row.get("right_ear_path", ""),
            )
            self._store_backbone_embeddings([image_key], images, left_ears, right_ears, image_type="upload")
        return self._projected_embedding(image_key, seek_code)

    def predict_seek_for_dataset_idx(self, idx):
        predictions = self.predict_seek_for_dataset_idxs([idx])
        return predictions.get(int(idx), "")

    def predict_seek_for_dataset_idxs(self, idxs, batch_size=64):
        self.load()
        results = {}
        missing = []
        for idx in [int(idx) for idx in idxs]:
            image_key = self._dataset_image_key(idx)
            predicted = self._feature(image_key).get("predicted_seek", "")
            if predicted:
                results[int(idx)] = predicted
            else:
                missing.append(int(idx))
        for start in range(0, len(missing), batch_size):
            batch_idxs = missing[start:start + batch_size]
            batch = self._stack_dataset_items(batch_idxs)
            image_keys = [self._dataset_image_key(idx) for idx in batch_idxs]
            self._store_concept_predictions(
                image_keys,
                batch["images"],
                batch["left_ears"],
                batch["right_ears"],
                image_type="dataset",
            )
            for idx, image_key in zip(batch_idxs, image_keys):
                results[int(idx)] = self._feature(image_key).get("predicted_seek", "")
        return results

    def predict_seek_for_upload(self, upload_row):
        self.load()
        image_key = self._upload_image_key(upload_row)
        predicted = self._feature(image_key).get("predicted_seek", "")
        if predicted:
            return predicted
        images, left_ears, right_ears = self._upload_tensors(
            upload_row["body_image_path"],
            upload_row.get("left_ear_path", ""),
            upload_row.get("right_ear_path", ""),
        )
        self._store_concept_predictions([image_key], images, left_ears, right_ears, image_type="upload")
        return self._feature(image_key).get("predicted_seek", "")

    def warm_dataset_query_features(self, idx, seek_code=None):
        predicted = self.predict_seek_for_dataset_idx(idx)
        self._dataset_projected_embeddings([idx], [seek_code or predicted])
        return predicted

    def warm_upload_query_features(self, upload_row, seek_code=None):
        predicted = self.predict_seek_for_upload(upload_row)
        self._upload_projected_embedding(upload_row, seek_code or predicted)
        return predicted

    def warm_visible_queries(self, rows, top_k=100, prediction_batch_size=64, rank_batch_size=4) -> dict[str, int]:
        rows = rows.copy()
        result = {"predicted": 0, "ranked": 0, "skipped": 0}
        dataset_rows = rows[
            (rows["query_id"].astype(str).str.startswith("dataset:"))
            & (rows["predicted_seek"].astype(str) == "")
        ]
        if not dataset_rows.empty:
            idxs = dataset_rows["idx"].astype(int).head(prediction_batch_size).tolist()
            predictions = self.predict_seek_for_dataset_idxs(idxs, batch_size=prediction_batch_size)
            for idx, predicted in predictions.items():
                self.state.log_query_prediction(f"dataset:{idx}", predicted)
                result["predicted"] += 1

        uploads = self.state.uploads_dataframe()
        upload_rows = rows[
            (rows["query_id"].astype(str).str.startswith("upload:"))
            & (rows["predicted_seek"].astype(str) == "")
        ]
        for query_id in upload_rows["query_id"].astype(str).head(prediction_batch_size).tolist():
            matches = uploads[uploads["query_id"] == query_id]
            if matches.empty:
                result["skipped"] += 1
                continue
            predicted = self.predict_seek_for_upload(matches.iloc[-1].to_dict())
            self.state.log_query_prediction(query_id, predicted)
            result["predicted"] += 1

        refreshed = self.catalog.all_queue_rows(self.state)
        gallery_version = self.state.gallery_version()
        for _, row in refreshed.iterrows():
            if result["ranked"] >= rank_batch_size:
                break
            query_id = str(row["query_id"])
            seek_code = str(row.get("predicted_seek", ""))
            if not seek_code:
                result["skipped"] += 1
                continue
            if self.state.cached_rank(query_id, seek_code, top_k, gallery_version=gallery_version) is not None:
                result["skipped"] += 1
                continue
            if query_id.startswith("dataset:"):
                ranking = self.rank_dataset_query(int(row["idx"]), seek_code, top_k=top_k)
            else:
                matches = uploads[uploads["query_id"] == query_id]
                if matches.empty:
                    result["skipped"] += 1
                    continue
                ranking = self.rank_upload_query(matches.iloc[-1].to_dict(), seek_code, top_k=top_k)
            self.state.log_rank_cache(query_id, seek_code, top_k, ranking)
            result["ranked"] += 1
        return result

    def refresh_gallery_cache(self, force=False, batch_size=64):
        self.load()
        version = self.state.gallery_version()
        if self.gallery_cache is not None and not force and self.gallery_cache.version == version:
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
                image_keys,
                batch["images"],
                batch["left_ears"],
                batch["right_ears"],
                image_type="dataset",
            )
            self._store_concept_predictions(
                image_keys,
                batch["images"],
                batch["left_ears"],
                batch["right_ears"],
                image_type="dataset",
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

        self.gallery_cache = GalleryCache(
            version=version,
            idxs=idxs,
            embeddings=gallery_embeddings,
            metadata=metadata,
        )
        return self.gallery_cache

    def update_gallery_cache_for_idxs(self, idxs, batch_size=64):
        self.gallery_cache = None
        return self.refresh_gallery_cache(force=True, batch_size=batch_size)

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

