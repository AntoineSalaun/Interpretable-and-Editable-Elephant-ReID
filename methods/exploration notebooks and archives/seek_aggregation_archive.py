"""
Archived SEEK aggregation helpers.

These functions used to live on the active `SEEK` class and were called by
inference paths through `aggregate_seeks=True`. Active experiments now keep
SEEK codes unaggregated, so this file preserves the old implementation for
analysis notebooks without keeping it in the runtime path.
"""

from pathlib import Path
import json
import os

import pandas as pd
import torch

from seek_code import SEEK


REPO_ROOT = Path(__file__).resolve().parents[2]
ELEPHANT4AFRICA_DATA_ROOT = REPO_ROOT / "data" / "Elephant4Africa"


DEFAULT_AGGREGATION_RULES = {
    "sex": {"cutoff": 0.9334337305774126, "fraction": 0.7151849388533961},
    "age": {"cutoff": 0.9514760451994138, "fraction": 0.5510405823660589},
    "tusks": {"cutoff": 0.4369025581005079, "fraction": 1.8715870244928936},
    "ear_most_prominent": {"cutoff": 0.8346892883784236, "fraction": 2.432201122842605},
    "ear_least_prominent": {"cutoff": 0.3902627093863492, "fraction": 2.0680981200569604},
    "extremes": {"cutoff": 0.46972912427893604, "fraction": 0.8622955318459603},
}


DEFAULT_VOTE_RULES = {
    "sex": {"cutoff": 0.7238117020284947, "fraction": 3.2846153859098934},
    "age": {"cutoff": 0.6080374539126856, "fraction": 3.030614746927617},
    "tusks": {"cutoff": 0.7864963346562002, "fraction": 1.3312930166044628},
    "ear_most_prominent": {"cutoff": 0.2133936013358265, "fraction": 1.788490949619698},
    "ear_least_prominent": {"cutoff": 0.1001741505264265, "fraction": 1.6865733430829595},
    "extremes": {"cutoff": 0.9641272649686263, "fraction": 0.9361114173246684},
}


def aggregate_seek(SEEK_codes, ele_ids, rules=None):
    if rules is None:
        rules = DEFAULT_AGGREGATION_RULES

    def make_seek_df(seek_codes, elephant_ids):
        data, seeks = [], []
        for i in range(seek_codes.shape[0]):
            seek = SEEK(seek_codes[i])
            data.append(seek.categorical_tensor()[0].tolist())
            seeks.append(str(seek))

        df = pd.DataFrame(data, columns=SEEK.attribute_names)
        df["subj_seek_str"] = seeks
        df["ele_id"] = elephant_ids.cpu() if isinstance(elephant_ids, torch.Tensor) else elephant_ids
        df["encounter_id"] = 1
        return df

    df = make_seek_df(SEEK_codes, ele_ids)
    df_mapped = df.copy()
    for col in df.columns:
        if col in SEEK.mappings:
            max_idx = len(SEEK.mappings[col]) - 1
            df_mapped[col] = df[col].apply(lambda x: SEEK.mappings[col][x] if 0 <= x <= max_idx else None)

    unknown_tokens = {"age": "__"}

    def normalized_counts(series, attr):
        unknown = unknown_tokens.get(attr, "_")
        cleaned = series.fillna(unknown).replace({None: unknown})
        counts = cleaned.value_counts(normalize=True, dropna=False)
        return {str(key): round(float(value), 3) for key, value in counts.items()}

    def generate_seek_code(row):
        return (
            f"{row['sex']}{row['age']}T{row['right_tusk']}{row['left_tusk']}"
            f"E{row['R_tear_1']}{row['R_hole_1']}{row['R_tear_2']}{row['R_hole_2']}-"
            f"{row['L_tear_1']}{row['L_hole_1']}{row['L_tear_2']}{row['L_hole_2']}X"
            f"{row['right_extreme']}{row['left_extreme']}S{row['ear_special']}{row['body_special']}"
        )

    elephant_seek_dict = {}
    for ele_id, group in df_mapped.groupby("ele_id"):
        elephant_seek_dict[ele_id] = {}
        for attr in SEEK.attribute_names:
            counts = normalized_counts(group[attr], attr)
            elephant_seek_dict[ele_id][attr] = aggregation_heuristic(attr, counts, rules)

    elephant_seek_df = pd.DataFrame.from_dict(elephant_seek_dict, orient="index")
    elephant_seek_df.index.name = "ele_id"

    merged_df = df_mapped.copy()
    for attr in SEEK.attribute_names:
        merged_df[attr] = merged_df["ele_id"].map(elephant_seek_df[attr])
    merged_df["SEEK_code"] = merged_df.apply(generate_seek_code, axis=1)

    ele_seek_list = [SEEK(seek).one_hot_encode() for seek in merged_df["SEEK_code"]]
    subject_seek_list = [SEEK(seek).one_hot_encode() for seek in df["subj_seek_str"]]

    device = ele_ids.device
    dtype = ele_ids.dtype
    subject_seek = torch.stack(subject_seek_list).to(device)
    ele_seek = torch.stack(ele_seek_list).to(device)
    ele_id = torch.as_tensor(merged_df["ele_id"].to_numpy(), dtype=dtype, device=device)
    return subject_seek, ele_seek, ele_id


def aggregation_heuristic(attr, counts, rules):
    attr_key = attr.lower()
    default = "__" if attr_key == "age" else "_"
    if not counts:
        return default

    def get_val(u, most, other, cfg):
        cutoff, fraction = cfg.get("cutoff", 0.8), cfg.get("fraction", 1)
        if counts.get(u, 0) > cutoff:
            return u
        if counts.get(other, 0) and counts.get(most, 0):
            if counts[other] > counts[most] * fraction:
                return other
            return most
        if counts.get(other, 0):
            return other
        if counts.get(most, 0):
            return most
        return u

    def get_tear(cfg):
        u, z = "_", "0"
        cutoff, fraction = cfg.get("cutoff", 0.9), cfg.get("fraction", 2)
        if counts.get(u, 0) > cutoff:
            return u
        other_sum = sum(val for key, val in counts.items() if key not in {u, z})
        if counts.get(z, 0) > other_sum * fraction:
            return z
        best = max((key for key in counts if key not in {u, z}), key=lambda k: counts[k], default=None)
        return best if best is not None else u

    if attr_key == "sex":
        return get_val("_", "B", "C", rules["sex"])
    if attr_key == "age":
        pick = get_val("__", "00", "20", rules["age"])
        return pick if pick in {"00", "20"} else "00"
    if attr_key in {"right_tusk", "left_tusk", "r_tusk", "l_tusk"}:
        return get_val("_", "1", "0", rules["tusks"])
    if "tear_1" in attr_key or "hole_1" in attr_key:
        return get_tear(rules["ear_most_prominent"])
    if "tear_2" in attr_key or "hole_2" in attr_key:
        return get_tear(rules["ear_least_prominent"])
    if "extreme" in attr_key:
        return get_val("1", "0", "_", rules["extremes"])
    if "special" in attr_key:
        return get_val("1", "0", "_", {"cutoff": 0.8, "fraction": 1.5})

    best = max(counts.items(), key=lambda item: item[1])[0]
    return best if best is not None else default


def aggregate_from_vote_to_SEEK(
    seek_counts_path=ELEPHANT4AFRICA_DATA_ROOT / "out_apr2" / "SEEK_dict.json",
    rules=None,
    sub_ele_pairs_path="/archive/vision/beery/animal_reid/datasets/elephants_zooniverse/ele_id_project/data/out_apr2/sub_ele_pairs.json",
    orinal_image_dict_path=ELEPHANT4AFRICA_DATA_ROOT / "out_apr2" / "image_dictonary_original.csv",
    export_path=None,
    drop_missing_ears=True,
):
    if rules is None:
        rules = DEFAULT_VOTE_RULES

    with open(seek_counts_path, "r", encoding="utf-8") as handle:
        seek_counts = json.load(handle)
    with open(sub_ele_pairs_path, "r", encoding="utf-8") as handle:
        sub_ele_pairs = json.load(handle)

    aggregated = {}
    for subject_id, features in seek_counts.items():
        aggregated[subject_id] = {}
        for feature, counts in features.items():
            total = sum(counts.values())
            if not total:
                aggregated[subject_id][feature] = None
                continue
            percentages = {val: round(count / total, 3) for val, count in counts.items()}
            aggregated[subject_id][feature] = aggregation_heuristic(feature, percentages, rules)

    image_seek_df = pd.DataFrame(aggregated).T
    image_seek_df.index.name = "subject_id"

    def build_seek(row):
        return (
            f"{row['Sex']}{row['Age']}T{row['R_tusk']}{row['L_tusk']}"
            f"E{row['R_tear_1']}{row['R_hole_1']}{row['R_tear_2']}{row['R_hole_2']}-"
            f"{row['L_tear_1']}{row['L_hole_1']}{row['L_tear_2']}{row['L_hole_2']}"
            f"X{row['R_extreme']}{row['L_extreme']}S{row['Special_ear']}{row['Special_body']}"
        )

    image_seek_df["SEEK"] = image_seek_df.apply(build_seek, axis=1).drop(image_seek_df.index[:3])
    image_seek_df["ele_id"] = image_seek_df.index.map(lambda idx: sub_ele_pairs.get(str(idx)))

    merged_df = pd.read_csv(orinal_image_dict_path)
    if "subject-SEEK" in merged_df.columns and "old-subject-SEEK" not in merged_df.columns:
        merged_df = merged_df.rename(columns={"subject-SEEK": "old-subject-SEEK"})
    else:
        raise ValueError("Input DataFrame must contain 'subject-SEEK' column but contains 'old-subject-SEEK' column.")

    image_seek_df = image_seek_df.copy()
    image_seek_df.index = image_seek_df.index.astype(int)
    merged_df = merged_df.merge(image_seek_df[["SEEK"]], left_on="subject_id", right_index=True, how="left")
    merged_df = merged_df.rename(columns={"SEEK": "subject-SEEK"})

    if "old-subject-SEEK" in merged_df.columns and "subject-SEEK" in merged_df.columns:
        cols = list(merged_df.columns)
        old_idx, new_idx = cols.index("old-subject-SEEK"), cols.index("subject-SEEK")
        if old_idx > new_idx:
            cols[new_idx], cols[old_idx] = cols[old_idx], cols[new_idx]
            merged_df = merged_df[cols]

    if drop_missing_ears and {"right_ear_path", "left_ear_path"}.issubset(merged_df.columns):
        right_missing = merged_df["right_ear_path"].isna() | merged_df["right_ear_path"].astype(str).str.lower().eq("nan")
        left_missing = merged_df["left_ear_path"].isna() | merged_df["left_ear_path"].astype(str).str.lower().eq("nan")
        valid_right = right_missing & merged_df["subject-SEEK"].notna()
        valid_left = left_missing & merged_df["subject-SEEK"].notna()
        merged_df.loc[valid_right, "subject-SEEK"] = (
            merged_df.loc[valid_right, "subject-SEEK"].astype(str).str.slice_replace(7, 11, "____")
        )
        merged_df.loc[valid_left, "subject-SEEK"] = (
            merged_df.loc[valid_left, "subject-SEEK"].astype(str).str.slice_replace(12, 16, "____")
        )

    if export_path:
        export_dir = os.path.dirname(export_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        target_df = merged_df if merged_df is not None else image_seek_df
        target_df.to_csv(export_path, index=False if merged_df is not None else True)

    return merged_df, image_seek_df
