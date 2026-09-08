# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "pillow",
#     "torch",
#     "torchvision",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import html
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    PACKAGE_DIR = ROOT / "marimo_interface"
    for path in (ROOT, PACKAGE_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from marimo_interface.config import DEFAULT_EXPERIMENT_DIR, RANK_CACHE_TOP_K, SQLITE_PATH
    from marimo_interface.data import DataCatalog
    from marimo_interface.model_service import ModelService
    from marimo_interface.preprocessing import preprocess_upload
    from marimo_interface.seek_utils import ATTRIBUTES, attrs_to_seek, is_valid_seek, normalize_seek, seek_to_attrs
    from marimo_interface.sqlite_state import SQLiteState

    return (
        ATTRIBUTES,
        DEFAULT_EXPERIMENT_DIR,
        DataCatalog,
        ModelService,
        Path,
        RANK_CACHE_TOP_K,
        SQLITE_PATH,
        SQLiteState,
        attrs_to_seek,
        hashlib,
        html,
        is_valid_seek,
        mo,
        normalize_seek,
        pd,
        preprocess_upload,
        seek_to_attrs,
    )


@app.cell
def _(DataCatalog, DEFAULT_EXPERIMENT_DIR, ModelService, SQLiteState):
    state = SQLiteState()
    catalog = DataCatalog()
    model = ModelService(catalog, state, experiment_dir=DEFAULT_EXPERIMENT_DIR)
    return catalog, model, state


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
        :root { color-scheme: light; }
        body, .marimo { background: #ffffff; color: #24292f; }
        .app-title { margin: 0 0 0.1rem 0; }
        .muted { color: #667085; font-size: 0.82rem; }
        .section-title {
            color: #303030;
            font-size: 0.92rem;
            font-weight: 750;
            margin: 0.1rem 0 0.35rem 0;
        }
        .seek-code {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.72rem;
            word-break: break-all;
        }
        .badge {
            display: inline-block;
            padding: 0.08rem 0.35rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            border: 1px solid rgba(31, 35, 40, 0.12);
        }
        .pending { background: #fff1bf; }
        .matched { background: #ccefd8; }
        .uncertain { background: #ded4ff; }
        .rejected { background: #ffd1d1; }
        .original { background: #f1f1f1; }
        .corrected { background: #eaf2f5; }
        .seek-match { background: #e6f1ea; border-radius: 3px; padding: 0 1px; }
        .seek-unknown { background: #f5edd8; border-radius: 3px; padding: 0 1px; }
        .seek-diff { background: #f4e5e5; border-radius: 3px; padding: 0 1px; }
        img { max-width: 100%; }
        </style>
        """
    )
    return


@app.cell
def _(
    ATTRIBUTES,
    Path,
    attrs_to_seek,
    html,
    is_valid_seek,
    mo,
    normalize_seek,
    pd,
    seek_to_attrs,
):
    BASE_SEEK = "B00T__E0000-0000X00S00"
    QUERY_STATUSES = ["pending", "matched", "uncertain", "rejected"]

    def safe_seek(code):
        try:
            return normalize_seek(code)
        except Exception:
            return str(code or "")

    def selected_table_row(table, fallback_df):
        value = table.value
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                return value.iloc[0].to_dict()
        elif isinstance(value, list) and value:
            return value[0]
        elif isinstance(value, dict):
            return value
        if fallback_df is None or fallback_df.empty:
            return None
        return fallback_df.iloc[0].to_dict()

    def status_badge(status):
        status = str(status or "pending")
        css = status if status in set(QUERY_STATUSES) else "pending"
        return mo.md(f"<span class='badge {css}'>{html.escape(status)}</span>")

    def correction_badge(corrected):
        css = "corrected" if corrected else "original"
        label = "corrected" if corrected else "original"
        return mo.md(f"<span class='badge {css}'>{label}</span>")

    def readable_time(value):
        if value is None or str(value).strip() == "" or str(value).lower() == "nan":
            return ""
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return str(value)
        return parsed.strftime("%b %-d, %Y %H:%M")

    def image_triplet(catalog, paths, height=260):
        items = []
        for label, key in (("body crop", "body"), ("left ear", "left_ear"), ("right ear", "right_ear")):
            image = catalog.open_image(paths.get(key))
            if image is None:
                continue
            items.append(mo.vstack([
                mo.image(image, height=height, style={"object-fit": "contain"}),
                mo.md(f"<div class='muted' style='text-align:center'>{label}</div>"),
            ], gap=0.15))
        if not items:
            return mo.md("<div class='muted'>No image crop is available.</div>")
        return mo.hstack(items, widths=[2, 1, 1][:len(items)], align="start", gap=0.75)

    def query_display_paths(catalog, meta):
        if meta.get("display_paths") is not None:
            display_paths = meta["display_paths"]
            if any(Path(str(path)).exists() for path in display_paths.values() if path):
                return display_paths
        return meta["paths"]

    def make_seek_widgets(initial_seek):
        try:
            attrs = seek_to_attrs(initial_seek)
        except Exception:
            attrs = seek_to_attrs(BASE_SEEK)
        controls = {}
        rows = []
        short_labels = {
            "right_tusk": "R tusk",
            "left_tusk": "L tusk",
            "right_extreme": "R ext",
            "left_extreme": "L ext",
            "ear_special": "ear sp.",
            "body_special": "body sp.",
        }
        for row_i in range(2):
            row_items = []
            for attr in ATTRIBUTES[row_i * 8:(row_i + 1) * 8]:
                value = attrs[attr.name] if attrs[attr.name] in attr.choices else attr.choices[0]
                control = mo.ui.dropdown(
                    attr.choices,
                    value=value,
                    label=short_labels.get(attr.name, attr.name.replace("_", " ")),
                    full_width=True,
                )
                controls[attr.name] = control
                row_items.append(control)
            rows.append(mo.hstack(row_items, widths="equal", gap=0.25))
        raw = mo.ui.text(value=safe_seek(initial_seek), label="Raw SEEK", full_width=True)
        return controls, raw, mo.vstack([
            mo.md("<div class='section-title'>SEEK editor</div>"),
            *rows,
            raw,
        ], gap=0.35)

    def seek_widgets_value(controls, raw):
        categorical = attrs_to_seek({name: control.value for name, control in controls.items()})
        code = raw.value.strip() or categorical
        valid = is_valid_seek(code)
        return normalize_seek(code) if valid else categorical, valid

    def highlighted_seek(query_seek, candidate_seek):
        query_seek = safe_seek(query_seek)
        candidate_seek = safe_seek(candidate_seek)
        pieces = []
        for i in range(max(len(query_seek), len(candidate_seek))):
            q = query_seek[i] if i < len(query_seek) else ""
            c = candidate_seek[i] if i < len(candidate_seek) else ""
            if q == c:
                css = "seek-match"
            elif q == "_" or c == "_":
                css = "seek-unknown"
            else:
                css = "seek-diff"
            pieces.append(f"<span class='{css}'>{html.escape(c or ' ')}</span>")
        return mo.md("<div class='seek-code'>" + "".join(pieces) + "</div>")

    def filter_query_rows(queue, statuses, subject, sighting, image):
        result = queue.copy()
        if statuses:
            result = result[result["status"].isin(statuses)]
        if subject.strip():
            result = result[result["subject_id"].astype(str) == subject.strip()]
        if sighting.strip():
            result = result[result["encounter_id"].astype(str) == sighting.strip()]
        if image.strip():
            result = result[result["image_id"].astype(str) == image.strip()]
        return result.reset_index(drop=True)

    def random_sighting_group_sort(df):
        if "encounter_id" not in df.columns or df.empty:
            return df
        result = df.copy()
        result["_sighting_order"] = result["encounter_id"].astype(str).map(
            lambda value: hashlib.md5(value.encode("utf-8")).hexdigest()
        )
        secondary = "image_id" if "image_id" in result.columns else "subject_id"
        return (
            result.sort_values(["_sighting_order", "encounter_id", secondary], kind="mergesort")
            .drop(columns=["_sighting_order"])
            .reset_index(drop=True)
        )

    def filter_gallery_rows(gallery, ele, subject, sighting):
        result = gallery.copy()
        if ele.strip():
            result = result[result["ele_id"].astype(str) == ele.strip()]
        if subject.strip():
            result = result[result["subject_id"].astype(str) == subject.strip()]
        if sighting.strip():
            result = result[result["encounter_id"].astype(str) == sighting.strip()]
        return result.reset_index(drop=True)

    def selected_query_metadata(catalog, state, query_id):
        if str(query_id).startswith("dataset:"):
            idx = int(str(query_id).split(":", 1)[1])
            return catalog.metadata_for_idx(idx, state=state, item_type="query")
        uploads = state.uploads_dataframe()
        row = uploads[uploads["query_id"] == query_id].iloc[-1]
        parent = Path(str(row["raw_image_path"])).parent
        return {
            "idx": None,
            "query_id": row["query_id"],
            "subject_id": "",
            "ele_id": "",
            "encounter_id": "",
            "subject_seek": "",
            "ele_seek": "",
            "current_ele_seek": "",
            "current_query_seek": state.latest_query_seek(row["query_id"]),
            "picture_time": row["timestamp"],
            "season": "upload",
            "paths": {
                "raw": row["raw_image_path"],
                "body": row["body_image_path"],
                "left_ear": row["left_ear_path"],
                "right_ear": row["right_ear_path"],
            },
            "display_paths": {
                "raw": row["raw_image_path"],
                "body": str(parent / "display" / "body_rgb.jpg"),
                "left_ear": str(parent / "display" / "left_ear_rgb.jpg"),
                "right_ear": str(parent / "display" / "right_ear_rgb.jpg"),
            },
            "upload_row": row.to_dict(),
        }

    def run_ranking(model, state, query_id, meta, seek_code, top_k):
        cached = state.cached_rank(query_id, seek_code, top_k)
        if cached is not None:
            return cached, "cached"
        if str(query_id).startswith("dataset:"):
            ranking = model.rank_dataset_query(meta["idx"], seek_code, top_k=top_k)
        else:
            ranking = model.rank_upload_query(meta["upload_row"], seek_code, top_k=top_k)
        state.log_ranking(query_id, seek_code, top_k, ranking)
        state.log_rank_cache(query_id, seek_code, top_k, ranking)
        return ranking, "computed"

    def gallery_affected_idxs(catalog, state, meta, scope_code):
        gallery = catalog.gallery_dataframe(state)
        if scope_code == "image":
            affected = gallery[gallery["subject_id"].astype(str) == str(meta["subject_id"])]
        elif scope_code == "elephant":
            affected = gallery[gallery["ele_id"].astype(str) == str(meta["ele_id"])]
        elif scope_code == "sighting":
            affected = gallery[gallery["encounter_id"].astype(str) == str(meta["encounter_id"])]
        else:
            affected = gallery[
                (gallery["ele_id"].astype(str) == str(meta["ele_id"]))
                & (gallery["encounter_id"].astype(str) == str(meta["encounter_id"]))
            ]
        return affected["idx"].astype(int).tolist()

    return (
        BASE_SEEK,
        QUERY_STATUSES,
        correction_badge,
        filter_gallery_rows,
        filter_query_rows,
        gallery_affected_idxs,
        highlighted_seek,
        image_triplet,
        make_seek_widgets,
        query_display_paths,
        random_sighting_group_sort,
        readable_time,
        run_ranking,
        seek_widgets_value,
        selected_query_metadata,
        selected_table_row,
        status_badge,
    )


@app.cell
def _(mo):
    tab_selector = mo.ui.tabs(
        {"Query Review": "", "Query Queue": "", "Gallery": ""},
        value="Query Review",
        lazy=True,
        label="View",
    )
    mo.vstack([
        mo.md("<h2 class='app-title'>Elephant Re-ID Expert Interface</h2>"),
        tab_selector,
    ], gap=0.2)
    return (tab_selector,)


@app.cell
def _(QUERY_STATUSES, mo):
    review_status_filter = mo.ui.multiselect(
        QUERY_STATUSES,
        value=QUERY_STATUSES,
        label="Status",
        full_width=True,
    )
    review_subject_filter = mo.ui.text(placeholder="exact", label="Subject ID", debounce=False, full_width=True)
    review_sighting_filter = mo.ui.text(placeholder="exact", label="Sighting ID", debounce=False, full_width=True)
    review_image_filter = mo.ui.text(placeholder="exact", label="Image ID", debounce=False, full_width=True)
    review_filters_ui = mo.hstack(
        [review_status_filter, review_subject_filter, review_sighting_filter, review_image_filter],
        widths="equal",
    )
    return (
        review_filters_ui,
        review_image_filter,
        review_sighting_filter,
        review_status_filter,
        review_subject_filter,
    )


@app.cell
def _(QUERY_STATUSES, mo):
    queue_status_filter = mo.ui.multiselect(
        QUERY_STATUSES,
        value=QUERY_STATUSES,
        label="Status",
        full_width=True,
    )
    queue_subject_filter = mo.ui.text(placeholder="exact", label="Subject ID", debounce=False, full_width=True)
    queue_sighting_filter = mo.ui.text(placeholder="exact", label="Sighting ID", debounce=False, full_width=True)
    queue_image_filter = mo.ui.text(placeholder="exact", label="Image ID", debounce=False, full_width=True)
    queue_filters_ui = mo.hstack(
        [queue_status_filter, queue_subject_filter, queue_sighting_filter, queue_image_filter],
        widths="equal",
    )
    upload_widget = mo.ui.file(filetypes=[".jpg", ".jpeg", ".png"], multiple=False, label="Upload query image")
    upload_button = mo.ui.run_button(label="Add upload", kind="success")
    warm_queue_button = mo.ui.run_button(label="Warm visible cache", kind="neutral")
    return (
        queue_filters_ui,
        queue_image_filter,
        queue_sighting_filter,
        queue_status_filter,
        queue_subject_filter,
        upload_button,
        upload_widget,
        warm_queue_button,
    )


@app.cell
def _(mo):
    gallery_ele_filter = mo.ui.text(placeholder="exact", label="Elephant ID", debounce=False, full_width=True)
    gallery_subject_filter = mo.ui.text(placeholder="exact", label="Image ID", debounce=False, full_width=True)
    gallery_sighting_filter = mo.ui.text(placeholder="exact", label="Sighting ID", debounce=False, full_width=True)
    gallery_filters_ui = mo.hstack(
        [gallery_ele_filter, gallery_subject_filter, gallery_sighting_filter],
        widths="equal",
    )
    return (
        gallery_ele_filter,
        gallery_filters_ui,
        gallery_sighting_filter,
        gallery_subject_filter,
    )


@app.cell
def _(
    catalog,
    filter_query_rows,
    queue_image_filter,
    random_sighting_group_sort,
    queue_sighting_filter,
    queue_status_filter,
    queue_subject_filter,
    review_image_filter,
    review_sighting_filter,
    review_status_filter,
    review_subject_filter,
    state,
):
    all_queue = catalog.all_queue_rows(state)
    review_filtered = filter_query_rows(
        all_queue,
        review_status_filter.value,
        review_subject_filter.value,
        review_sighting_filter.value,
        review_image_filter.value,
    )
    queue_filtered = random_sighting_group_sort(filter_query_rows(
        all_queue,
        queue_status_filter.value,
        queue_subject_filter.value,
        queue_sighting_filter.value,
        queue_image_filter.value,
    ))
    return all_queue, queue_filtered, review_filtered


@app.cell
def _(catalog, filter_gallery_rows, gallery_ele_filter, gallery_sighting_filter, gallery_subject_filter, state):
    all_gallery = catalog.gallery_dataframe(state)
    gallery_filtered = filter_gallery_rows(
        all_gallery,
        gallery_ele_filter.value,
        gallery_subject_filter.value,
        gallery_sighting_filter.value,
    )
    return all_gallery, gallery_filtered


@app.cell
def _(mo, review_filtered):
    review_table_df = review_filtered[[
        "query_id", "image_id", "subject_id", "encounter_id", "status",
        "selected_ele_id", "predicted_seek", "corrected_seek",
    ]] if not review_filtered.empty else review_filtered
    review_table = mo.ui.table(
        review_table_df,
        selection="single",
        pagination=True,
        page_size=25,
        show_data_types=False,
        show_download=False,
        label="Database",
    )
    return review_table, review_table_df


@app.cell
def _(mo, queue_filtered):
    queue_table_df = queue_filtered[[
        "query_id", "image_id", "subject_id", "encounter_id", "status",
        "selected_ele_id", "predicted_seek", "corrected_seek",
    ]] if not queue_filtered.empty else queue_filtered
    queue_table = mo.ui.table(
        queue_table_df,
        selection="single",
        pagination=True,
        page_size=25,
        show_data_types=False,
        show_download=False,
        label="Database",
    )
    return queue_table, queue_table_df


@app.cell
def _(gallery_filtered, mo, readable_time):
    gallery_table_df = gallery_filtered[[
        "idx", "subject_id", "ele_id", "encounter_id", "picture_time",
        "subject_seek", "ele_seek", "current_ele_seek", "corrected_seek",
    ]].copy() if not gallery_filtered.empty else gallery_filtered
    if not gallery_table_df.empty and "picture_time" in gallery_table_df:
        gallery_table_df["picture_time"] = gallery_table_df["picture_time"].map(readable_time)
    gallery_table = mo.ui.table(
        gallery_table_df,
        selection="single",
        pagination=True,
        page_size=25,
        show_data_types=False,
        show_download=False,
        label="Database",
    )
    return gallery_table, gallery_table_df


@app.cell
def _(catalog, review_table, review_table_df, selected_query_metadata, selected_table_row, state):
    review_selected = selected_table_row(review_table, review_table_df)
    if review_selected is None:
        review_meta = {}
        review_query_id = ""
        review_current_seek = "B00T__E0000-0000X00S00"
        review_cached_prediction = ""
    else:
        review_query_id = str(review_selected["query_id"])
        review_meta = selected_query_metadata(catalog, state, review_query_id)
        review_cached_prediction = state.query_predictions().get(review_query_id, "")
        review_current_seek = review_meta.get("current_query_seek") or review_cached_prediction or "B00T__E0000-0000X00S00"
    return review_cached_prediction, review_current_seek, review_meta, review_query_id, review_selected


@app.cell
def _(catalog, queue_table, queue_table_df, selected_query_metadata, selected_table_row, state):
    queue_selected = selected_table_row(queue_table, queue_table_df)
    if queue_selected is None:
        queue_meta = {}
        queue_query_id = ""
        queue_current_seek = "B00T__E0000-0000X00S00"
    else:
        queue_query_id = str(queue_selected["query_id"])
        queue_meta = selected_query_metadata(catalog, state, queue_query_id)
        queue_current_seek = state.latest_query_seek(queue_query_id) or queue_selected.get("predicted_seek") or "B00T__E0000-0000X00S00"
    return queue_current_seek, queue_meta, queue_query_id, queue_selected


@app.cell
def _(catalog, gallery_table, gallery_table_df, selected_table_row, state):
    gallery_selected = selected_table_row(gallery_table, gallery_table_df)
    if gallery_selected is None:
        gallery_meta = {}
        gallery_current_seek = "B00T__E0000-0000X00S00"
    else:
        gallery_meta = catalog.metadata_for_idx(int(gallery_selected["idx"]), state=state, item_type="gallery")
        gallery_current_seek = gallery_meta["current_ele_seek"]
    return gallery_current_seek, gallery_meta, gallery_selected


@app.cell
def _(make_seek_widgets, review_current_seek):
    review_seek_controls, review_seek_raw, review_seek_editor_ui = make_seek_widgets(review_current_seek)
    return review_seek_controls, review_seek_editor_ui, review_seek_raw


@app.cell
def _(make_seek_widgets, queue_current_seek):
    queue_seek_controls, queue_seek_raw, queue_seek_editor_ui = make_seek_widgets(queue_current_seek)
    return queue_seek_controls, queue_seek_editor_ui, queue_seek_raw


@app.cell
def _(gallery_current_seek, make_seek_widgets):
    gallery_seek_controls, gallery_seek_raw, gallery_seek_editor_ui = make_seek_widgets(gallery_current_seek)
    return gallery_seek_controls, gallery_seek_editor_ui, gallery_seek_raw


@app.cell
def _(review_seek_controls, review_seek_raw, seek_widgets_value):
    review_edited_seek, review_seek_valid = seek_widgets_value(review_seek_controls, review_seek_raw)
    return review_edited_seek, review_seek_valid


@app.cell
def _(queue_seek_controls, queue_seek_raw, seek_widgets_value):
    queue_edited_seek, queue_seek_valid = seek_widgets_value(queue_seek_controls, queue_seek_raw)
    return queue_edited_seek, queue_seek_valid


@app.cell
def _(gallery_seek_controls, gallery_seek_raw, seek_widgets_value):
    gallery_edited_seek, gallery_seek_valid = seek_widgets_value(gallery_seek_controls, gallery_seek_raw)
    return gallery_edited_seek, gallery_seek_valid


@app.cell
def _(all_queue, mo, review_meta, review_seek_valid):
    review_sighting_id = str(review_meta.get("encounter_id", ""))
    review_scope_options = {"image-level": "image"}
    if review_sighting_id and len(all_queue[all_queue["encounter_id"].astype(str) == review_sighting_id]) > 1:
        review_scope_options["sighting-level"] = "elephant_sighting"
    review_scope = mo.ui.dropdown(
        list(review_scope_options.keys()),
        value="sighting-level" if "sighting-level" in review_scope_options else "image-level",
        label="Correction scope",
        full_width=True,
    )
    review_apply_button = mo.ui.run_button(
        label="Apply correction",
        kind="success",
        disabled=not review_seek_valid,
        full_width=True,
    )
    review_compute_button = mo.ui.run_button(label="Compute candidates", kind="neutral", full_width=True)
    review_identity_input = mo.ui.text(label="Elephant ID", placeholder="Elephant ID", full_width=True)
    review_choose_button = mo.ui.run_button(
        label="Choose",
        kind="success",
        full_width=True,
    )
    review_uncertain_button = mo.ui.run_button(label="Uncertain", kind="warn", full_width=True)
    review_reject_button = mo.ui.run_button(label="Reject", kind="danger", full_width=True)
    return (
        review_apply_button,
        review_choose_button,
        review_compute_button,
        review_identity_input,
        review_reject_button,
        review_scope,
        review_scope_options,
        review_uncertain_button,
    )


@app.cell
def _(mo, queue_seek_valid):
    queue_apply_button = mo.ui.run_button(
        label="Apply query SEEK",
        kind="success",
        disabled=not queue_seek_valid,
    )
    return (queue_apply_button,)


@app.cell
def _(gallery_seek_valid, mo):
    gallery_scope = mo.ui.dropdown(
        {
            "this image only": "image",
            "all images of this elephant": "elephant",
            "all images in this sighting": "sighting",
            "all images of this elephant in this sighting": "elephant_sighting",
        },
        value="all images of this elephant in this sighting",
        label="Apply correction",
        full_width=True,
    )
    gallery_save_button = mo.ui.run_button(
        label="Save gallery SEEK",
        kind="success",
        disabled=not gallery_seek_valid,
    )
    gallery_hide_button = mo.ui.run_button(label="Delete from UI gallery", kind="danger")
    return gallery_hide_button, gallery_save_button, gallery_scope


@app.cell
def _(
    RANK_CACHE_TOP_K,
    all_queue,
    catalog,
    model,
    review_apply_button,
    review_choose_button,
    review_compute_button,
    review_edited_seek,
    review_identity_input,
    review_meta,
    review_query_id,
    review_reject_button,
    review_scope,
    review_scope_options,
    review_selected,
    review_uncertain_button,
    run_ranking,
    selected_query_metadata,
    state,
):
    review_messages = []
    review_ranking_source = "cached predicted SEEK"
    review_predicted_for_rank = state.query_predictions().get(review_query_id, "") if review_query_id else ""
    review_ranking = []
    if review_query_id:
        review_ranking = (
            state.cached_rank(review_query_id, review_edited_seek, RANK_CACHE_TOP_K)
            or state.cached_rank(review_query_id, review_predicted_for_rank, RANK_CACHE_TOP_K)
            or []
        )
    if review_apply_button.value and review_query_id:
        scope_code = review_scope_options[review_scope.value]
        targets = [review_query_id]
        if scope_code == "elephant_sighting":
            sighting_id = str(review_meta.get("encounter_id", ""))
            targets = all_queue[all_queue["encounter_id"].astype(str) == sighting_id]["query_id"].astype(str).tolist()
        for target_query_id in targets:
            target_meta = selected_query_metadata(catalog, state, target_query_id)
            state.log_seek_correction(
                item_type="query",
                query_id=target_query_id,
                subject_id=target_meta["subject_id"],
                ele_id=target_meta["ele_id"],
                encounter_id=target_meta["encounter_id"],
                original_subject_seek=target_meta["subject_seek"],
                original_ele_seek=target_meta["ele_seek"],
                predicted_seek=state.query_predictions().get(target_query_id, ""),
                corrected_seek=review_edited_seek,
                scope=scope_code,
            )
            if str(target_query_id).startswith("dataset:"):
                model.warm_dataset_query_features(target_meta["idx"], review_edited_seek)
            else:
                model.warm_upload_query_features(target_meta["upload_row"], review_edited_seek)
        review_ranking, review_ranking_source = run_ranking(
            model, state, review_query_id, review_meta, review_edited_seek, RANK_CACHE_TOP_K
        )
        review_messages.append("correction applied")
    if review_compute_button.value and review_query_id:
        if not review_predicted_for_rank:
            if str(review_query_id).startswith("dataset:"):
                predicted = model.predict_seek_for_dataset_idx(review_meta["idx"])
            else:
                predicted = model.predict_seek_for_upload(review_meta["upload_row"])
            state.log_query_prediction(review_query_id, predicted)
        review_ranking, review_ranking_source = run_ranking(
            model, state, review_query_id, review_meta, review_edited_seek, RANK_CACHE_TOP_K
        )
        review_messages.append("candidates computed")
    if review_choose_button.value and review_query_id and review_identity_input.value.strip():
        chosen_identity = review_identity_input.value.strip()
        state.log_decision(
            review_query_id,
            review_meta.get("subject_id", ""),
            "matched",
            selected_ele_id=chosen_identity,
        )
        review_messages.append(f"matched to {chosen_identity}")
    if review_uncertain_button.value and review_query_id:
        state.log_decision(review_query_id, review_meta.get("subject_id", ""), "uncertain")
        review_messages.append("marked uncertain")
    if review_reject_button.value and review_query_id:
        state.log_decision(review_query_id, review_meta.get("subject_id", ""), "rejected")
        review_messages.append("rejected")
    if review_selected is None:
        review_ranking = []
    return review_messages, review_ranking, review_ranking_source


@app.cell
def _(catalog, model, preprocess_upload, queue_apply_button, queue_edited_seek, queue_meta, queue_query_id, state, upload_button, upload_widget, warm_queue_button, queue_filtered):
    queue_messages = []
    if upload_button.value and upload_widget.value:
        uploaded = upload_widget.value[0] if isinstance(upload_widget.value, list) else upload_widget.value
        row = preprocess_upload(uploaded, state)
        queue_messages.append(f"added {row['query_id']}")
    if warm_queue_button.value and not queue_filtered.empty:
        result = model.warm_visible_queries(queue_filtered.head(25), top_k=100)
        queue_messages.append(
            f"predicted {result['predicted']} · ranked {result['ranked']} · skipped {result['skipped']}"
        )
    if queue_apply_button.value and queue_query_id:
        state.log_seek_correction(
            item_type="query",
            query_id=queue_query_id,
            subject_id=queue_meta.get("subject_id", ""),
            ele_id=queue_meta.get("ele_id", ""),
            encounter_id=queue_meta.get("encounter_id", ""),
            original_subject_seek=queue_meta.get("subject_seek", ""),
            original_ele_seek=queue_meta.get("ele_seek", ""),
            predicted_seek=state.query_predictions().get(queue_query_id, ""),
            corrected_seek=queue_edited_seek,
            scope="image",
        )
        queue_messages.append("query SEEK saved")
    return (queue_messages,)


@app.cell
def _(
    catalog,
    gallery_affected_idxs,
    gallery_edited_seek,
    gallery_hide_button,
    gallery_meta,
    gallery_save_button,
    gallery_scope,
    model,
    state,
):
    gallery_messages = []
    if gallery_save_button.value and gallery_meta:
        affected_idxs = gallery_affected_idxs(catalog, state, gallery_meta, gallery_scope.value)
        state.log_seek_correction(
            item_type="gallery",
            query_id="",
            subject_id=gallery_meta["subject_id"],
            ele_id=gallery_meta["ele_id"],
            encounter_id=gallery_meta["encounter_id"],
            original_subject_seek=gallery_meta["subject_seek"],
            original_ele_seek=gallery_meta["ele_seek"],
            predicted_seek="",
            corrected_seek=gallery_edited_seek,
            scope=gallery_scope.value,
        )
        model.update_gallery_cache_for_idxs(affected_idxs)
        gallery_messages.append(f"saved for {len(affected_idxs)} image(s)")
    if gallery_hide_button.value and gallery_meta:
        state.hide_gallery_image(gallery_meta["subject_id"], gallery_meta["encounter_id"])
        model.gallery_cache = None
        gallery_messages.append("hidden from gallery retrieval")
    return (gallery_messages,)


@app.cell
def _(mo, pd, review_ranking):
    if review_ranking:
        candidate_table_df = pd.DataFrame(review_ranking)[[
            "rank", "score", "ele_id", "subject_id", "encounter_id", "current_ele_seek", "idx",
        ]].copy()
        candidate_table_df["score"] = candidate_table_df["score"].map(lambda value: f"{float(value):.4f}")
    else:
        candidate_table_df = pd.DataFrame(columns=[
            "rank", "score", "ele_id", "subject_id", "encounter_id", "current_ele_seek", "idx",
        ])
    candidate_table = mo.ui.table(
        candidate_table_df,
        selection="single",
        pagination=True,
        page_size=10,
        show_data_types=False,
        show_download=False,
        label="Candidates",
    )
    return candidate_table, candidate_table_df


@app.cell
def _(candidate_table, candidate_table_df, selected_table_row):
    candidate_selected = selected_table_row(candidate_table, candidate_table_df)
    return (candidate_selected,)


@app.cell
def _(
    catalog,
    candidate_selected,
    correction_badge,
    gallery_current_seek,
    gallery_edited_seek,
    gallery_filters_ui,
    gallery_hide_button,
    gallery_meta,
    gallery_messages,
    gallery_save_button,
    gallery_scope,
    gallery_seek_editor_ui,
    gallery_selected,
    gallery_table,
    highlighted_seek,
    image_triplet,
    mo,
    query_display_paths,
    queue_apply_button,
    queue_edited_seek,
    queue_filters_ui,
    queue_messages,
    queue_meta,
    queue_seek_editor_ui,
    queue_selected,
    queue_table,
    review_apply_button,
    review_cached_prediction,
    review_choose_button,
    review_compute_button,
    review_edited_seek,
    review_filters_ui,
    review_identity_input,
    review_messages,
    review_meta,
    review_ranking_source,
    review_reject_button,
    review_scope,
    review_seek_editor_ui,
    review_selected,
    review_table,
    review_uncertain_button,
    status_badge,
    upload_button,
    upload_widget,
    warm_queue_button,
):
    review_message_ui = [mo.md(f"<span class='badge matched'>{m}</span>") for m in review_messages]
    queue_message_ui = [mo.md(f"<span class='badge matched'>{m}</span>") for m in queue_messages]
    gallery_message_ui = [mo.md(f"<span class='badge matched'>{m}</span>") for m in gallery_messages]

    if review_selected is None:
        review_content = mo.vstack([
            mo.md("<div class='section-title'>Filters</div>"),
            review_filters_ui,
            review_table,
            mo.md("No matching queries."),
        ])
    else:
        query_panel = mo.vstack([
            mo.md("<div class='section-title'>Query</div>"),
            mo.hstack([
                status_badge(review_selected.get("status", "pending")),
                mo.md(f"<span class='muted'>cached prediction</span> <span class='seek-code'>{review_cached_prediction or '-'}</span>"),
            ], justify="start"),
            image_triplet(catalog, query_display_paths(catalog, review_meta)),
            review_seek_editor_ui,
            mo.hstack([review_scope, review_apply_button, review_compute_button], widths=[1.3, 0.7, 0.7]),
            mo.accordion({
                "Cheat sheet": mo.vstack([
                    mo.md(f"true elephant ID: `{review_meta.get('ele_id') or '-'}`"),
                    mo.md(f"image-level SEEK: `{review_meta.get('subject_seek') or '-'}`"),
                    mo.md(f"sighting-level SEEK: `{review_meta.get('ele_seek') or '-'}`"),
                ])
            }),
            mo.md("<div class='section-title'>Decision</div>"),
            mo.hstack([review_uncertain_button, review_reject_button, review_identity_input, review_choose_button], widths=[0.7, 0.7, 1.2, 0.7]),
            *review_message_ui,
        ])
        candidate_bits = [
            mo.md("<div class='section-title'>Candidates</div>"),
            mo.md(f"<div class='muted'>source: {review_ranking_source}</div>"),
            candidate_table,
        ]
        if candidate_selected is not None and str(candidate_selected.get("idx", "")):
            candidate_paths = catalog.image_paths(catalog.row(int(candidate_selected["idx"])))
            candidate_bits.extend([
                mo.md(
                    f"<div class='section-title'>Candidate elephant {candidate_selected['ele_id']} "
                    f"· rank {candidate_selected['rank']} · score {candidate_selected['score']}</div>"
                ),
                highlighted_seek(review_edited_seek, candidate_selected["current_ele_seek"]),
                image_triplet(catalog, candidate_paths),
            ])
        candidate_panel = mo.vstack(candidate_bits)
        review_content = mo.vstack([
            mo.md("<div class='section-title'>Filters</div>"),
            review_filters_ui,
            review_table,
            mo.hstack([query_panel, candidate_panel], widths=[1.05, 1.25], align="start", gap=1.0),
        ], gap=0.65)

    if queue_selected is None:
        queue_content = mo.vstack([
            mo.md("<div class='section-title'>Upload</div>"),
            mo.hstack([upload_widget, upload_button], widths=[1, 0.18], align="end"),
            mo.md("<div class='section-title'>Filters</div>"),
            queue_filters_ui,
            queue_table,
            mo.md("No matching queries."),
            *queue_message_ui,
        ])
    else:
        queue_content = mo.vstack([
            mo.md("<div class='section-title'>Upload</div>"),
            mo.hstack([upload_widget, upload_button], widths=[1, 0.18], align="end"),
            mo.md("<div class='section-title'>Filters</div>"),
            queue_filters_ui,
            mo.hstack([queue_table, warm_queue_button], widths=[1, 0.18], align="start"),
            mo.hstack([
                mo.vstack([
                    mo.md(f"<div class='section-title'>Selected query {queue_selected['image_id']}</div>"),
                    status_badge(queue_selected.get("status", "pending")),
                    image_triplet(catalog, query_display_paths(catalog, queue_meta)),
                ]),
                mo.vstack([queue_seek_editor_ui, queue_apply_button, *queue_message_ui]),
            ], widths=[1.1, 1.0], align="start"),
        ], gap=0.65)

    if gallery_selected is None:
        gallery_content = mo.vstack([
            mo.md("<div class='section-title'>Filters</div>"),
            gallery_filters_ui,
            gallery_table,
            mo.md("No matching gallery images."),
            *gallery_message_ui,
        ])
    else:
        gallery_content = mo.vstack([
            mo.md("<div class='section-title'>Filters</div>"),
            gallery_filters_ui,
            gallery_table,
            mo.hstack([
                mo.vstack([
                    mo.md(
                        f"<div class='section-title'>Image {gallery_meta['subject_id']} · "
                        f"elephant {gallery_meta['ele_id']} · sighting {gallery_meta['encounter_id']}</div>"
                    ),
                    correction_badge(gallery_meta["corrected_seek"]),
                    image_triplet(catalog, query_display_paths(catalog, gallery_meta)),
                    mo.md(f"<span class='muted'>image-level</span> <span class='seek-code'>{gallery_meta['subject_seek']}</span>"),
                    mo.md(f"<span class='muted'>sighting-level</span> <span class='seek-code'>{gallery_meta['ele_seek']}</span>"),
                    mo.md(f"<span class='muted'>current</span> <span class='seek-code'>{gallery_current_seek}</span>"),
                ]),
                mo.vstack([gallery_seek_editor_ui, gallery_scope, mo.hstack([gallery_save_button, gallery_hide_button]), *gallery_message_ui]),
            ], widths=[1.1, 1.0], align="start"),
        ], gap=0.65)

    return gallery_content, queue_content, review_content


@app.cell
def _(
    DEFAULT_EXPERIMENT_DIR,
    SQLITE_PATH,
    gallery_content,
    mo,
    queue_content,
    review_content,
    state,
    tab_selector,
):
    cache_counts = state.cache_counts()
    content = {
        "Query Review": review_content,
        "Query Queue": queue_content,
        "Gallery": gallery_content,
    }[tab_selector.value]
    mo.vstack([
        mo.md(
            f"<div class='muted'>Projector checkpoint: <code>{DEFAULT_EXPERIMENT_DIR}</code> · "
            f"sqlite: <code>{SQLITE_PATH}</code> · "
            f"features {cache_counts['feature_cache']} · "
            f"projected {cache_counts['projected_embedding_cache']} · "
            f"ranks {cache_counts['rank_cache']}</div>"
        ),
        content,
    ], gap=0.6)
    return


if __name__ == "__main__":
    app.run()
