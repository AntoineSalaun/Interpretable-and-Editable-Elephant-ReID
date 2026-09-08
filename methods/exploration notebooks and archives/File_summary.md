# Methods Directory Summary

## .env
- Stores environment variables loaded by other scripts (e.g., `data_handler.py`). Sensitive values are not tracked in version control.

## backbone.py
- Defines the `Backbone` ArcFace embedding model that wraps vision backbones such as MegaDescriptor or MiewID and manages training/evaluation loops.

### Functions & Methods
- `Backbone.__init__(model_name="MegaDescriptor", with_ears=True, pretraining="savannah_elephants", lr=5e-6, experiment_code=None, print_every=5, num_classes=310)` → sets up the selected transformer backbone, optionally loads pretrained weights, freezes layers, and prepares optimizers. Expects `self.experiment_dir` to exist before writing logs.
- `Backbone.freeze()` / `Backbone.unfreeze()` → toggle gradient updates on the backbone weights. No inputs besides `self`; no return.
- `Backbone.forward(images, left_ears=None, right_ears=None)` → returns concatenated embeddings for main images and optional ear crops. Inputs: batched tensors shaped `(N, 3, 224, 224)`.
- `Backbone.epoch_pass(loader, training=True)` → runs one pass over a dataloader, computes ArcFace loss and recall@1. Returns `(epoch_loss: float, epoch_accuracy: float)`.
- `Backbone.train(train_loader, val_loader, num_epochs=10)` → unfreezes the model and performs multi-epoch training while logging metrics and storing the best weights. Returns `None`.
- `Backbone.test(train_loader, test_loader)` → evaluates retrieval metrics on held-out data and writes them to disk. Returns `None`.
- `Backbone.collect_embeddings(loader, **kwargs)` → gathers embeddings and labels for retrieval. Returns `(embeddings: torch.Tensor, labels: torch.Tensor)`.

## CBM_mara.ipynb
- Exploration notebook for CBM experiments on the Mara dataset. Holds interactive analysis cells rather than reusable functions.

## classifier.py
- Implements a simple linear (or optional MLP) classifier operating on 2304-d backbone embeddings.

### Functions & Methods
- `Classifier.__init__(num_classes, criterion=nn.CrossEntropyLoss(), lr=0.001, experiment_code=None, softmax=False, dropout=False, wd=0.0)` → builds the classification head, optionally loads weights, and configures optimizer/logging.
- `Classifier.compute_accuracy(logits, labels)` → returns `float` accuracy for the batch.
- `Classifier.epoch_pass(loader, backbone, training=True)` → runs forward/backward passes against a backbone; returns `(loss: float, accuracy: float)`.
- `Classifier.train(train_loader, val_loader, backbone, num_epochs=30)` → initializes weights, trains for `num_epochs`, saves best parameters, and logs history.
- `Classifier.test(test_loader, backbone)` → evaluates loss/accuracy on a test loader and writes accuracy to disk.
- `Classifier.freeze()` / `Classifier.unfreeze()` → toggles gradients for the classifier layers.

## concept_head_tunneled.py
- Provides loss helpers, the concept-head training loop, and several architectural variants that predict SEEK codes from embeddings.

### Free Functions
- `categorical_CE_loss(output, labels)` → sums per-attribute cross-entropy losses over SEEK parts. Inputs: logits tensor `(N, 63)` and one-hot labels `(N, 63)`; output: scalar loss tensor.

### ConceptHeadTunneled
- `ConceptHeadTunneled.__init__(lr=1e-5, loss=categorical_CE_loss, experiment_code=None, reset_weights=True, layer=None, pretraining='concept_w')` → configures concept head training, loads optional weights, and builds experiment directories.
- `ConceptHeadTunneled.accuracy_fn(predicted, labels)` → returns dict of per-attribute accuracies plus summary metrics using closest valid SEEK predictions.
- `ConceptHeadTunneled.epoch_pass(loader, backbone, training=True, predict_ele_SEEK=False)` → processes a loader, applies optional intervention, and returns `(loss: float, accuracies: dict)`.
- `ConceptHeadTunneled.train(train_loader, val_loader, backbone, num_epochs=10, predict_ele_SEEK=False)` → jointly optimizes the concept head (and optionally backbone) and logs history.
- `ConceptHeadTunneled.test(test_loader, backbone, predict_ele_SEEK=False)` → writes per-attribute accuracies to disk and returns `(loss, accuracy_dict)`.
- `ConceptHeadTunneled.retrieval_test(backbone_for_concepts, train_loader, test_loader, intervention_fns)` → evaluates retrieval under multiple interventions and distances; returns mapping of results.
- `ConceptHeadTunneled.freeze()` / `ConceptHeadTunneled.unfreeze()` → manage train/eval modes for the concept head submodules.
- `ConceptHeadTunneled.collect_embeddings(loader, backbone, intervention_fn=None, aggregate_seeks=False, aggregate_rules=None)` → produces predicted SEEK codes (optionally aggregated) alongside elephant labels.

### Concept Head Architectures
- `ThreeHeadNN.forward(embeddings)` → maps concatenated (whole, left, right) embeddings to 63-d logits via three linear heads; returns `(N, 63)` tensor.
- `MultiHeadNN.forward(embeddings)` → deeper per-region heads plus per-attribute prediction dictionaries; returns `(N, 63)` tensor. Includes `save_model(path)` / `load_model(path)` for checkpointing.
- `CrossedHeadNN.forward(embeddings)` → uses concatenated region features for ear attributes; returns `(N, 63)` tensor plus `save_model`/`load_model` helpers.

## data_handler.py
- Contains the `EleHandler` dataset for loading images, ears, metadata, and SEEK encodings from CSV dictionaries. Also provides split utilities and visualization helpers.

### Functions & Methods
- `EleHandler.__init__(dictonary_path=None, dataset_type='zooniverse', subset=None)` → loads dataset metadata, applies optional subset filtering, and builds label mappings.
- `EleHandler.__len__()` → returns dataset size.
- `EleHandler.__getitem__(idx)` → returns tuple `(preprocessed_image, subject_id, ele_id_label, identified_flag, subject_seek_one_hot, ele_seek_one_hot, left_ear_tensor, right_ear_tensor, subject_seek_str, ele_seek_str, idx)`.
- `EleHandler._load_ear(ear_path)` → helper returning ear tensor or zero image when missing.
- `EleHandler.get_original_image(idx=None, subject_id=None)` → retrieves raw RGB PIL image for inspection.
- `EleHandler.print_image(idx, print_with_transform=True)` → visualizes raw, preprocessed, and ear crops plus logs metadata; returns `(image, label_dict)`.
- `EleHandler.get_IDI_indices()` / `EleHandler.get_IDU_indices()` → indices for specific season subsets.
- `EleHandler.split_along_encounters(split_sizes=[0.7,0.15,0.15], hour_delta=1, identification=None)` → hour-based chronological split; returns `(train_idx, val_idx, test_idx)`.
- `EleHandler.split_parallel_to_encounters(split_sizes=[0.5,0.5])` → encounter-wise split ensuring single-encounter elephants stay in train.
- `EleHandler.split_perpendicular_to_elephants_and_encounters(split_sizes=[0.5,0.5], hour_delta=0.2)` → balanced split maintaining elephant coverage.

## debug_experiments.ipynb
- Notebook used for ad-hoc debugging of experiments; not structured as callable code.

## evaluating_CBM.ipynb
- Notebook summarizing evaluation results and plots for the CBM pipeline.

## exp_baselines.py
- CLI entry point orchestrating various experiment presets (baseline retrieval, concept-head training, projector training). Defines argument parsing and chooses training routines based on `--experiment` and `--dataset` flags. No reusable functions beyond the script-level control flow.

## exploration notebooks and archives/
- Directory containing historical or exploratory notebooks/artefacts not meant for direct import.

## find_best_agg_strategy.py
- Uses Optuna to tune SEEK aggregation heuristics for improved retrieval recall.

### Functions
- `setup(dictonary_path)` → prepares dataset loaders, retrieval helper, backbone, and concept head; returns dict with reusable components.
- `_suggest_rules(trial)` → samples rule hyperparameters from predefined bounds.
- `objective(trial)` → aggregates SEEK codes with sampled rules and returns recall@1 for optimization.
- `tune_rules_for_recall1(n_trials=100, seed=0)` → runs Optuna study and returns `(best_rules: dict, best_recall: float)`.

## new_concept_agg.ipynb
- Experiment notebook focused on SEEK aggregation strategies and diagnostics.

## projector.py
- Implements the `Projector` module that maps concept logits back to embedding space and blends them with backbone features while training with metric-learning losses.

### Functions & Methods
- `Projector.__init__(loss_type='ArcFace', lr=5e-5, margin=0.5, scale=64, experiment_code=None, reset_weights=True, intervention_fn=None, alpha=0.5, print_every=5, network_type='small')` → builds projection network, loads optional weights, configures losses/optimizers, and sets up logging.
- `Projector.compute_recall_at_k(embeddings, labels, k=1)` → helper returning recall@k via a new `Retrieval` instance.
- `Projector.epoch_pass(loader, backbone_for_concepts, backbone, ch, training=True, intervention_fn=None)` → trains/evaluates one epoch, applying intervention to concept logits, and returns `(loss, recall)`.
- `Projector.collect_embeddings(loader, backbone, backbone_for_concepts, concept_head, intervention_fn=None, aggregate_seeks=False, aggregate_rules=None)` → gathers edited embeddings and labels, with optional SEEK aggregation.
- `Projector.train(train_loader, val_loader, backbone_for_concepts, backbone, concept_head, num_epochs=10)` → optimizes projector (and any unfrozen backbone layers), tracks best validation recall, and saves checkpoints.
- `Projector.make_logger(log_path, level=logging.INFO, overwrite=False)` → configures file/stream logging; returns `logging.Logger`.
- `Projector.test(backbone_for_concepts, backbone, concept_head, train_loader, test_loader, intervention_fns)` → evaluates retrieval metrics across interventions and aggregation settings; returns dict of results.
- `Projector.freeze()` / `Projector.unfreeze()` → toggle gradient requirements for the projector network.

## retrieval.py
- Contains the `Retrieval` helper class for computing similarity matrices, recall metrics, and visualization utilities (plots, t-SNE, qualitative match views).

### Functions & Methods
- `Retrieval.__init__(experiment_code=None)` → creates experiment directory for logs/plots.
- `Retrieval.similarity_matrix(query_embeddings, gallery_embeddings=None, distance='cosine_sim')` → computes cosine or SEEK-based similarity; returns `(N, M)` tensor.
- `Retrieval.compute_recall_at_k(similarity_matrix, query_labels, gallery_labels, k)` → returns recall value for specified `k`.
- `Retrieval.one_out_retrieval(model, loader, backbone_for_concepts=None, backbone=None, print=False, ba=None, ch=None, intervention_fn=None)` → collects embeddings via model, computes cosine recall for several `k`s, and optionally prints summary.
- `Retrieval.evaluate_model(model, train_loader, test_loader, model_to_evaluate=None, ba=None, ch=None, backbone_for_concepts=None, show_tsne=False, show_plot=False, show_matches=False, intervention_fn=None, aggregate_gallery_seeks=True, print_results=True, distance='cosine_sim')` → orchestrates gallery/query embedding creation and returns test recall dictionary.
- `Retrieval.plot_accuracy_vs_samples(similarity_matrix, query_labels, gallery_labels, ks)` → saves accuracy vs. sample count visualization.
- `Retrieval.visualize_matches(similarity_matrix, query_labels, gallery_labels, retrieval_dataset, query_subset, gallery_subset, n_vis=4, k=600)` → generates qualitative retrieval panels.
- `Retrieval.make_tsne(embeddings, labels)` → saves a t-SNE scatter plot of embeddings.

## seek_code.py
- Defines the SEEK code representation, encoding/decoding utilities, aggregation heuristics, intervention policies, and custom distance metrics used across the project.

### SEEK Class
- `SEEK.__init__(code)` → accepts SEEK string or `(63,)` one-hot tensor and initializes attribute fields.
- `SEEK.initialize_from_string(code)` / `SEEK.initialize_from_one_hot(one_hot_vector)` → internal helpers for parsing input representations. Raise on invalid lengths.
- `SEEK.__getitem__(index)` → returns one-hot tensor for the requested attribute index.
- `SEEK.one_hot_encode()` → concatenates attribute one-hots into `(63,)` tensor.
- `SEEK.__str__()` → reconstructs canonical SEEK string.
- `SEEK.closest_valid_one_hot(prob_vector)` → converts logits/probabilities `(N, 63)` to the nearest valid one-hot predictions.
- `SEEK.categorical_tensor()` → returns tuple `(full_tensor, whole_indices_tensor, left_indices_tensor, right_indices_tensor)` of categorical indices.
- `SEEK.batch_categorical(labels)` → vectorized version accepting iterable of SEEK strings; returns stacked tensors on CUDA.
- `SEEK.separate_one_hot(labels_1h)` → splits one-hot tensor(s) into whole, right, and left attribute groups.
- `SEEK.rescontruct_from_separated_prob_vectors(whole_vector, left_vector, right_vector)` → reassembles separated logits into `(N, 63)` tensor.
- `SEEK.test_separate_and_reconstruct()` → diagnostic that verifies split/reconstruction correctness.
- `SEEK.aggregate_seek(SEEK_codes, ele_ids, rules=None)` → aggregates SEEK predictions per elephant using heuristic rules; returns `(subject_seek_tensor, aggregated_ele_seek_tensor, ele_ids_tensor)`.
- `SEEK.aggregation_heuristic(attr, counts, rules)` → selects the most plausible value for an attribute given frequency counts and thresholds.
- `SEEK.aggregate_from_vote_to_SEEK(..., export_path=None, drop_missing_ears=True)` → merges crowdsourced vote counts into SEEK codes, optionally updating CSV exports.
- `SEEK.perfect_correction(...)`, `SEEK.oracle_correction(...)`, `SEEK.correct_or_soft(...)`, `SEEK.correct_or_hard_at_prob(...)`, `SEEK.correct_or_hard(...)`, `SEEK.hard(...)`, `SEEK.perfect(...)` → intervention policies returning edited SEEK tensors.
- `SEEK.distance(...)`, `SEEK.distance_2(...)`, `SEEK.distance_3(...)` → custom distance metrics between SEEK-coded elephants.

### Module-Level Tests
- `test_seek()` and `test_distance()` → runnable sanity checks when executing the module directly.

## wandb/
- Directory placeholder for Weights & Biases run metadata and cached artifacts; not committed to source control.
