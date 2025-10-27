

from seek_code import SEEK
from backbone import Backbone
from retrieval import Retrieval
from projector import Projector
from data_handler import EleHandler
from concept_head_tunneled import ConceptHeadTunneled, categorical_CE_loss, CrossedHeadNN, MultiHeadNN

from pathlib import Path
from torch.utils.data import DataLoader, Subset
import os,json
import pandas as pd


def aggregate_from_vote_to_SEEK(
    seek_counts_path = '/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/SEEK_dict.json',
    rules=None,
    sub_ele_pairs_path= '/archive/vision/beery/animal_reid/datasets/elephants_zooniverse/ele_id_project/data/out_apr2/sub_ele_pairs.json',
    orinal_image_dict_path  ='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_original.csv',
    export_path=None,
    drop_missing_ears=True
):
    """Read SEEK vote counts from JSON, apply the supplied rules, and return aggregated codes.

    The function loads the nested counts structure from ``seek_counts_path``, resolves the
    winning value for every SEEK character using the provided ``rules`` dictionary, then
    stitches the components back together into full SEEK strings. When ``orinal_image_dict`` is given,
    the new codes are merged in place so downstream comparisons stay easy. Optional export
    and ear-removal hooks mirror the original notebook workflow without forcing callers to
    repeat that boilerplate.
    """

    with open(seek_counts_path, 'r') as handle: seek_counts = json.load(handle)
    with open(sub_ele_pairs_path, 'r') as handle: sub_ele_pairs = json.load(handle)

    def _get_val(p_counts, u, most_likely, other, cutoff=0.8, fraction=1):
        if u in p_counts and p_counts[u] > cutoff:
            return u

        if other in p_counts and most_likely in p_counts:
            if p_counts[other] > (p_counts[most_likely] * fraction):
                return other
            return most_likely

        if other in p_counts:
            return other

        if most_likely in p_counts:
            return most_likely

        #print('None of the expected values were found in dictionary:', p_counts, ' u is ', u, ' most likely is ', most_likely, ' and other is ', other)
        return u

    def _get_tear_hole_val(p, cutoff=0.9, frac=2):
        u = '_'
        z = '0'

        if u in p and p[u] > cutoff:
            return u

        other_sum = sum(value for key, value in p.items() if key not in {z, u})

        if z in p and p[z] > (other_sum * frac):
            return z

        max_key = None
        max_value = -1
        for key, value in p.items():
            if key not in {z, u} and value > max_value:
                max_key = key
                max_value = value

        return max_key if max_key is not None else u

    def _age_only_rule(pct):
        v = _get_val(pct, u='__', most_likely='00', other='20', cutoff=rules['age']['cutoff'], fraction=rules['age']['fraction'])
        return v if v in {'00', '20'} else '00'

    feature_rules = {
        'Sex': lambda pct: _get_val(pct, u='_', most_likely='B', other='C', cutoff=rules['sex']['cutoff'], fraction=rules['sex']['fraction']),
        'Age': _age_only_rule,
        'R_tusk': lambda pct: _get_val(pct, u='_', most_likely='1', other='0', cutoff=rules['tusks']['cutoff'], fraction=rules['tusks']['fraction']),
        'L_tusk': lambda pct: _get_val(pct, u='_', most_likely='1', other='0', cutoff=rules['tusks']['cutoff'], fraction=rules['tusks']['fraction']),
        'R_tear_1': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_most_prominent']['cutoff'], frac=rules['ear_most_prominent']['fraction']),
        'R_hole_1': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_most_prominent']['cutoff'], frac=rules['ear_most_prominent']['fraction']),
        'L_tear_1': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_most_prominent']['cutoff'], frac=rules['ear_most_prominent']['fraction']),
        'L_hole_1': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_most_prominent']['cutoff'], frac=rules['ear_most_prominent']['fraction']),
        'R_tear_2': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_least_prominent']['cutoff'], frac=rules['ear_least_prominent']['fraction']),
        'R_hole_2': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_least_prominent']['cutoff'], frac=rules['ear_least_prominent']['fraction']),
        'L_tear_2': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_least_prominent']['cutoff'], frac=rules['ear_least_prominent']['fraction']),
        'L_hole_2': lambda pct: _get_tear_hole_val(pct, cutoff=rules['ear_least_prominent']['cutoff'], frac=rules['ear_least_prominent']['fraction']),
        'R_extreme': lambda pct: _get_val(pct, u='1', most_likely='0', other='_', cutoff=rules['extremes']['cutoff'], fraction=rules['extremes']['fraction']),
        'L_extreme': lambda pct: _get_val(pct, u='1', most_likely='0', other='_', cutoff=rules['extremes']['cutoff'], fraction=rules['extremes']['fraction']),
        'Special_ear': lambda pct: _get_val(pct, u='1', most_likely='0', other='_', cutoff=0.8, fraction=1.5),
        'Special_body': lambda pct: _get_val(pct, u='1', most_likely='0', other='_', cutoff=0.8, fraction=1.5),
    }

    aggregated = {}
    for subject_id, features in seek_counts.items():
        aggregated[subject_id] = {}
        for feature, counts in features.items():
            total = sum(counts.values())
            if not total:
                aggregated[subject_id][feature] = None
                continue
            percentages = {val: round(count / total, 3) for val, count in counts.items()}
            handler = feature_rules.get(feature)
            if handler is None:
                aggregated[subject_id][feature] = max(percentages.items(), key=lambda item: item[1])[0]
            else:
                aggregated[subject_id][feature] = handler(percentages)

    image_seek_df = pd.DataFrame(aggregated).T
    image_seek_df.index.name = 'subject_id'


    def build_seek(row: pd.Series) -> str:
        seek_str = str(row['Sex']) + str(row['Age']) + 'T' + str(row['R_tusk']) + str(row['L_tusk']) + 'E' + str(row['R_tear_1']) + str(row['R_hole_1']) + str(row['R_tear_2']) + str(row['R_hole_2']) + '-' + str(row['L_tear_1']) + str(row['L_hole_1']) + str(row['L_tear_2']) + str(row['L_hole_2']) + 'X' + str(row['R_extreme']) + str(row['L_extreme']) + 'S' + str(row['Special_ear']) + str(row['Special_body'])
        return seek_str

    image_seek_df['SEEK'] = image_seek_df.apply(build_seek, axis=1).drop(image_seek_df.index[:3])

    image_seek_df['ele_id'] = image_seek_df.index.map(lambda idx: sub_ele_pairs.get(str(idx)))


    merged_df = pd.read_csv(orinal_image_dict_path)

    if 'subject-SEEK' in merged_df.columns and 'old-subject-SEEK' not in merged_df.columns:
        merged_df = merged_df.rename(columns={'subject-SEEK': 'old-subject-SEEK'})
    else: raise ValueError("Input DataFrame must contain 'subject-SEEK' column but contains 'old-subject-SEEK' column.")

    # ensure index types match and merge only the aggregated SEEK column
    image_seek_df = image_seek_df.copy()
    image_seek_df.index = image_seek_df.index.astype(int)
    merged_df = merged_df.merge(image_seek_df[['SEEK']], left_on='subject_id', right_index=True, how='left')
    merged_df = merged_df.rename(columns={'SEEK': 'subject-SEEK'})

    if 'old-subject-SEEK' in merged_df.columns and 'subject-SEEK' in merged_df.columns:
        cols = list(merged_df.columns)
        old_idx, new_idx = cols.index('old-subject-SEEK'), cols.index('subject-SEEK')
        if old_idx > new_idx:
            cols[new_idx], cols[old_idx] = cols[old_idx], cols[new_idx]
            merged_df = merged_df[cols]

    if drop_missing_ears and {'right_ear_path', 'left_ear_path'}.issubset(merged_df.columns):
        def remove_ears(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            right_missing = df['right_ear_path'].isna() | df['right_ear_path'].astype(str).str.lower().eq('nan')
            left_missing = df['left_ear_path'].isna() | df['left_ear_path'].astype(str).str.lower().eq('nan')
            valid_right = right_missing & df['subject-SEEK'].notna()
            valid_left = left_missing & df['subject-SEEK'].notna()
            df.loc[valid_right, 'subject-SEEK'] = (
                df.loc[valid_right, 'subject-SEEK'].astype(str).str.slice_replace(7, 11, '____')
            )
            df.loc[valid_left, 'subject-SEEK'] = (
                df.loc[valid_left, 'subject-SEEK'].astype(str).str.slice_replace(12, 16, '____')
            )
            return df

        merged_df = remove_ears(merged_df)

    if export_path:
        export_dir = os.path.dirname(export_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        target_df = merged_df if merged_df is not None else image_seek_df
        target_df.to_csv(export_path, index=False if merged_df is not None else True)

    return merged_df, image_seek_df


def test_compute_traditional_seek_codes_basic():
    traditional_rules = {
    'sex': {'cutoff': 0.8, 'fraction': 2},
    'age': {'cutoff': 0.8, 'fraction': 1},
    'tusks': {'cutoff': 0.8, 'fraction': 1},
    'ear_most_prominent': {'cutoff': 0.9, 'fraction': 2},
    'ear_least_prominent': {'cutoff': 0.95, 'fraction': 2.5},
    'extremes': {'cutoff': 0.8, 'fraction': 1.5}
    }

    merged_df_mv,image_seek_df_mv = aggregate_from_vote_to_SEEK(
        rules=traditional_rules,
        export_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_traditional_rules.csv',
        drop_missing_ears=False
    )

    count, dist = 0, 0
    for subject_id, row in merged_df_mv.iterrows():
        old_code = SEEK(row['old-subject-SEEK'])
        new_code = SEEK(row['subject-SEEK'])
        x = SEEK.distance_3(old_code, new_code)
        dist += x
        count +=1 

    print('The average distance from the old-school way to our new function is ', dist / count )

    merged_df_mv,image_seek_df_mv = aggregate_from_vote_to_SEEK(
    rules=traditional_rules,
    export_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_traditional_rules_without_ears.csv',
    drop_missing_ears=True
    )

    count, dist = 0, 0
    for subject_id, row in merged_df_mv.iterrows():
        old_code = SEEK(row['old-subject-SEEK'])
        new_code = SEEK(row['subject-SEEK'])
        x = SEEK.distance_3(old_code, new_code)
        dist += x
        count +=1 

    print('If we detect the ears, the distance goes up slightly ', dist / count )

    majority_vote_rules = {
    'sex': {'cutoff': 0.51, 'fraction': 1},
    'age': {'cutoff': 0.51, 'fraction': 1},
    'tusks': {'cutoff': 0.51, 'fraction': 1},
    'ear_most_prominent': {'cutoff': 0.51,'fraction': 1},
    'ear_least_prominent': {'cutoff': 0.51,'fraction': 1},
    'extremes': {'cutoff': 0.51, 'fraction': 1}
    }

    merged_df_mv,image_seek_df_mv = aggregate_from_vote_to_SEEK(
    rules=majority_vote_rules,
    export_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_traditional_rules.csv',
    drop_missing_ears=True
    )

    count, dist = 0, 0
    for subject_id, row in merged_df_mv.iterrows():
        old_code = SEEK(row['old-subject-SEEK'])
        new_code = SEEK(row['subject-SEEK'])
        x = SEEK.distance_3(old_code, new_code)
        dist += x
        count +=1 
        #print('old SEEK:', old_code, ' new SEEK:', new_code,' distance is' , x)
    
    print('And if we switch to majority rule, the distance is pretty different : ', dist / count )

test_compute_traditional_seek_codes_basic()


traditional_rules = {
'sex': {'cutoff': 0.8, 'fraction': 2},
'age': {'cutoff': 0.8, 'fraction': 1},
'tusks': {'cutoff': 0.8, 'fraction': 1},
'ear_most_prominent': {'cutoff': 0.9, 'fraction': 2},
'ear_least_prominent': {'cutoff': 0.95, 'fraction': 2.5},
'extremes': {'cutoff': 0.8, 'fraction': 1.5}
}


def retrieval_vs_rule(rules):
    _, _ = aggregate_from_vote_to_SEEK(
    rules=traditional_rules,
    export_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_temp.csv',
    drop_missing_ears=True
    )


    dataset = EleHandler(subset='IDI_6', dictonary_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_temp.csv')
    train_indices, test_indices = dataset.split_perpendicular_to_elephants_and_encounters(split_sizes=[0.5,0.5], hour_delta = 0.2)

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=True)

    code ='aggregation-test0'
    r = Retrieval()

    MD_for_concepts = Backbone(model_name="MegaDescriptor", pretraining="backbone_for_concepts_w", experiment_code=code)
    concept_head = ConceptHeadTunneled(loss=categorical_CE_loss, experiment_code = code, layer = CrossedHeadNN(), reset_weights=False, pretraining='concept_w')

    query_embeddings, query_labels = concept_head.collect_embeddings(test_loader, backbone=MD_for_concepts, intervention_fn=SEEK.perfect_correction, aggregate_seeks=False)
    gallery_embeddings, gallery_labels = concept_head.collect_embeddings(train_loader, backbone=MD_for_concepts, intervention_fn=SEEK.perfect_correction, aggregate_seeks=True)
    test_similarity_matrix = r.similarity_matrix(query_embeddings, gallery_embeddings, distance='seek_homemade_3')

    # Compute and print recalls
    test_recalls = {k: r.compute_recall_at_k(similarity_matrix=test_similarity_matrix, query_labels=query_labels, gallery_labels=gallery_labels, k=k) for k in [1,5,10,20,100]}

    return test_recalls


def distant_from_agg_to_orcale(rules):
    merged_df_mv,image_seek_df_mv = aggregate_from_vote_to_SEEK(
    rules=rules,
    export_path='/data/vision/beery/scratch/antoine/CBM_reid/data_processing/data/out_apr2/image_dictonary_temp.csv',
    drop_missing_ears=True
    )

    count, dist = 0, 0
    for subject_id, row in merged_df_mv.iterrows():
        old_code = SEEK(row['subject-SEEK'])
        new_code = SEEK(row['ele-SEEK'])
        x = SEEK.distance_3(old_code, new_code) 
        dist += x
        count +=1 

    return dist / count




avg_dist = distant_from_agg_to_orcale(traditional_rules)
print('The average distance from the oracle to our traditional aggregation function is ', avg_dist)



import math
import optuna
from optuna.samplers import TPESampler

# ---- Your objective (already provided) is assumed to be in scope:
# def distant_from_agg_to_orcale(rules): ...
# and aggregate_from_vote_to_SEEK, SEEK etc. are importable.

# Sensible default ranges per field (feel free to tweak)
_DEFAULT_BOUNDS = {
    "sex":                {"cutoff": (0.2, 0.99),  "fraction": (0.5, 3.5)},
    "age":                {"cutoff": (0.2, 0.99),  "fraction": (0.5, 3.5)},
    "tusks":              {"cutoff": (0.2, 0.99),  "fraction": (0.5, 3.5)},
    "ear_most_prominent": {"cutoff": (0.2, 0.995), "fraction": (0.5, 3.5)},
    "ear_least_prominent":{"cutoff": (0.1, 0.999), "fraction": (0.5, 3.5)},
    "extremes":           {"cutoff": (0.2, 0.99),  "fraction": (0.5, 3.5)},
}

def tune_rules(
    base_rules=  traditional_rules,
    *,
    bounds=None,
    n_trials: int = 150,
    seed: int = 0,
    storage: str | None = None,   # e.g., "sqlite:///tuning.db" to persist
    study_name: str | None = None
):
    """
    Minimze distant_from_agg_to_orcale(rules) over (cutoff, fraction) for each field.

    Args
    ----
    base_rules: dict | None
        Optional starting rules; fields not in bounds won't be tuned.
        If None, uses your current defaults (from the prompt).
    bounds: dict | None
        Optional per-field bounds overriding _DEFAULT_BOUNDS.
        Example:
          {
            "sex": {"cutoff": (0.6, 0.95), "fraction": (1.0, 3.0)},
            "ear_least_prominent": {"cutoff": (0.9, 0.999), "fraction": (1.0, 4.0)},
          }
    n_trials: int
        Number of trials to run.
    seed: int
        Seed for reproducibility.
    storage: str | None
        Optuna storage URI to persist results (e.g., SQLite).
    study_name: str | None
        Name of the study when using persistent storage.

    Returns
    -------
    best_rules: dict
        The tuned rules dict.
    best_value: float
        The best objective value found.
    study: optuna.study.Study
        The Optuna study object (for plots, importance, etc.).
    """
        

    tune_bounds = _DEFAULT_BOUNDS.copy()
    if bounds:
        # shallow merge: only override provided fields
        for k, v in bounds.items():
            tune_bounds[k] = {**tune_bounds.get(k, {}), **v}

    fields_to_tune = list(tune_bounds.keys())

    def objective(trial: optuna.trial.Trial) -> float:
        # Start from base_rules; replace tuned fields with suggestions
        rules = {k: dict(v) for k, v in base_rules.items()}

        for field in fields_to_tune:
            # Skip fields not present in base_rules (safety)
            if field not in rules:
                rules[field] = {}

            lo_c, hi_c = tune_bounds[field]["cutoff"]
            lo_f, hi_f = tune_bounds[field]["fraction"]

            # Cutoffs between 0 and 1 — search linearly near top end
            rules[field]["cutoff"] = trial.suggest_float(f"{field}.cutoff", lo_c, hi_c)

            # Fractions are positive, can vary a lot → log scale helps
            rules[field]["fraction"] = trial.suggest_float(
                f"{field}.fraction", lo_f, hi_f, log=True
            )

        # Evaluate your pipeline
        try:
            val = distant_from_agg_to_orcale(rules)
        except Exception as e:
            # If something goes wrong in data/aggregation, nuke this trial
            trial.set_user_attr("error", str(e))
            return -math.inf  # we're maximizing

        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return -math.inf

        return float(val)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed, n_startup_trials=min(20, max(5, n_trials // 10))),
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Build the best rules dict back from base + best params
    best = base_rules.copy()
    for field in fields_to_tune:
        if f"{field}.cutoff" in study.best_params:
            best[field] = {
                "cutoff": study.best_params[f"{field}.cutoff"],
                "fraction": study.best_params[f"{field}.fraction"],
            }

    return best, study.best_value, study

best_rules, best_val, study = tune_rules(n_trials=10000, seed=42)

print("Best value:", best_val)
print("Best rules:")
for k, v in best_rules.items():
    print(k, v)


from optuna.importance import get_param_importances
imps = get_param_importances(study)
print("\nParameter importances:")
for k, v in imps.items():
    print(f"{k}: {v:.3f}")


retrieval_vs_rule(best_rules)



