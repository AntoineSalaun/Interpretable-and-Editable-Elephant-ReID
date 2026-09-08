from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from embedding_atlas.streamlit import embedding_atlas


THIS_DIR = Path(__file__).resolve().parent
MAX_IMAGES_TO_SHOW = 12
COLOR_BUCKET_FIELD = "ele_id"
EXPORT_DIR = THIS_DIR.parent / "embedding_atlas" / "exported_visualization"
EXPORT_CANVAS = (1800, 1400)
EXPORT_MARGIN = 48.0
EXPORT_POINT_RADIUS = 6.0
EXPORT_POINT_ALPHA = 0.82
EXPORT_CATEGORICAL_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#edc948",
    "#b07aa1",
    "#76b7b2",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]
EXPORT_PALETTE = {
    "C00": "#1f77b4",
    "C01": "#ff7f0e",
    "C02": "#2ca02c",
    "C03": "#d62728",
    "C04": "#9467bd",
    "C05": "#8c564b",
    "C06": "#e377c2",
    "C07": "#7f7f7f",
    "C08": "#bcbd22",
    "C09": "#17becf",
}
CORRECTION_OPTIONS = [
    ("00", "0% correction"),
    ("25", "25% correction"),
    ("50", "50% correction"),
    ("75", "75% correction"),
    ("100", "100% correction"),
    ("oracle", "Oracle correction"),
]
SEEK_ATTRIBUTE_SLICES = [
    ("sex", 0, 1, "Sex"),
    ("age", 1, 3, "Age"),
    ("right_tusk", 4, 5, "Right Tusk"),
    ("left_tusk", 5, 6, "Left Tusk"),
    ("R_tear_1", 7, 8, "Right Tear 1"),
    ("R_hole_1", 8, 9, "Right Hole 1"),
    ("R_tear_2", 9, 10, "Right Tear 2"),
    ("R_hole_2", 10, 11, "Right Hole 2"),
    ("L_tear_1", 12, 13, "Left Tear 1"),
    ("L_hole_1", 13, 14, "Left Hole 1"),
    ("L_tear_2", 14, 15, "Left Tear 2"),
    ("L_hole_2", 15, 16, "Left Hole 2"),
    ("right_extreme", 17, 18, "Right Extreme"),
    ("left_extreme", 18, 19, "Left Extreme"),
    ("special_block", 20, 23, "Special Block"),
]
COLOR_OPTIONS = [
    ("ele_id", "Elephant Bucket"),
    ("compact_elephant_id", "Elephant Identity"),
]
for seek_prefix, seek_label in (("subject_seek", "Subject SEEK"), ("ele_seek", "Elephant SEEK")):
    for attribute_name, _, _, attribute_label in SEEK_ATTRIBUTE_SLICES:
        COLOR_OPTIONS.append((f"{seek_prefix}_{attribute_name}", f"{seek_label}: {attribute_label}"))


@st.cache_data(show_spinner=False)
def load_data(correction_code: str) -> tuple[pd.DataFrame, dict]:
    data_path = THIS_DIR / "data" / f"mara_embedding_{correction_code}_top50.parquet"
    info_path = THIS_DIR / "data" / f"mara_embedding_{correction_code}_top50.build.json"
    data_frame = enrich_seek_columns(pd.read_parquet(data_path))
    build_info = json.loads(info_path.read_text())
    return data_frame, build_info


def enrich_seek_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = data_frame.copy()
    for seek_prefix in ("subject_seek", "ele_seek"):
        seek_values = enriched[seek_prefix].fillna("").astype(str)
        for attribute_name, start_idx, end_idx, _ in SEEK_ATTRIBUTE_SLICES:
            column_name = f"{seek_prefix}_{attribute_name}"
            enriched[column_name] = seek_values.str.slice(start_idx, end_idx).replace("", pd.NA)
    return enriched


def compute_viewport(data_frame: pd.DataFrame) -> dict:
    x_min = float(data_frame["tsne_x"].min())
    x_max = float(data_frame["tsne_x"].max())
    y_min = float(data_frame["tsne_y"].min())
    y_max = float(data_frame["tsne_y"].max())

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    radius = max(
        abs(x_min - center_x),
        abs(x_max - center_x),
        abs(y_min - center_y),
        abs(y_max - center_y),
        1e-6,
    )

    return {
        "x": center_x,
        "y": center_y,
        "scale": 0.95 / radius,
    }


def build_initial_state(data_frame: pd.DataFrame, color_field: str) -> dict:
    return {
        "layoutStates": {
            "list": {
                "showTable": False,
                "showCharts": False,
                "showEmbedding": True,
            }
        },
        "charts": {
            "1": {
                "type": "embedding",
                "title": "Embedding",
                "data": {
                    "x": "tsne_x",
                    "y": "tsne_y",
                    "text": "atlas_text",
                    "category": color_field,
                },
                "mode": "points",
                "viewport": compute_viewport(data_frame),
            },
            "2": {
                "type": "predicates",
                "title": "SQL Predicates",
            },
            "3": {
                "type": "instances",
                "title": "Instances",
            },
        }
    }


def query_selection(data_frame: pd.DataFrame, predicate: str | None) -> pd.DataFrame:
    if not predicate:
        return data_frame.iloc[0:0].copy()

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("atlas_df", data_frame)
        return connection.execute(f"SELECT * FROM atlas_df WHERE {predicate}").fetch_df()
    finally:
        connection.close()


def pdf_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


def build_export_color_map(data_frame: pd.DataFrame, color_field: str) -> dict[str, str]:
    if color_field == "ele_id":
        return dict(EXPORT_PALETTE)

    values = (
        data_frame[color_field]
        .dropna()
        .astype(str)
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    return {
        value: EXPORT_CATEGORICAL_PALETTE[index % len(EXPORT_CATEGORICAL_PALETTE)]
        for index, value in enumerate(values)
    }


def circle_path(cx: float, cy: float, radius: float) -> list[str]:
    handle = radius * 0.552284749831
    return [
        f"{pdf_number(cx + radius)} {pdf_number(cy)} m",
        (
            f"{pdf_number(cx + radius)} {pdf_number(cy + handle)} "
            f"{pdf_number(cx + handle)} {pdf_number(cy + radius)} "
            f"{pdf_number(cx)} {pdf_number(cy + radius)} c"
        ),
        (
            f"{pdf_number(cx - handle)} {pdf_number(cy + radius)} "
            f"{pdf_number(cx - radius)} {pdf_number(cy + handle)} "
            f"{pdf_number(cx - radius)} {pdf_number(cy)} c"
        ),
        (
            f"{pdf_number(cx - radius)} {pdf_number(cy - handle)} "
            f"{pdf_number(cx - handle)} {pdf_number(cy - radius)} "
            f"{pdf_number(cx)} {pdf_number(cy - radius)} c"
        ),
        (
            f"{pdf_number(cx + handle)} {pdf_number(cy - radius)} "
            f"{pdf_number(cx + radius)} {pdf_number(cy - handle)} "
            f"{pdf_number(cx + radius)} {pdf_number(cy)} c"
        ),
    ]


def write_vector_pdf(path: Path, width: float, height: float, content_stream: str) -> None:
    content_bytes = content_stream.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pdf_number(width)} {pdf_number(height)}] "
            "/Contents 4 0 R /Resources << /ExtGState << /GS1 5 0 R >> >> >>"
        ).encode("ascii"),
        b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"\nendstream",
        f"<< /Type /ExtGState /ca {EXPORT_POINT_ALPHA} /CA {EXPORT_POINT_ALPHA} >>".encode("ascii"),
    ]

    pdf_bytes = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{object_index} 0 obj\n".encode("ascii"))
        pdf_bytes.extend(obj)
        pdf_bytes.extend(b"\nendobj\n")

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf_bytes.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    pdf_bytes.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    path.write_bytes(pdf_bytes)


def export_embedding_pdf(data_frame: pd.DataFrame, build_info: dict, color_field: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    width = float(EXPORT_CANVAS[0])
    height = float(EXPORT_CANVAS[1])
    left = EXPORT_MARGIN + EXPORT_POINT_RADIUS
    right = width - EXPORT_MARGIN - EXPORT_POINT_RADIUS
    bottom = EXPORT_MARGIN + EXPORT_POINT_RADIUS
    top = height - EXPORT_MARGIN - EXPORT_POINT_RADIUS
    plot_width = right - left
    plot_height = top - bottom

    x_min = float(data_frame["tsne_x"].min())
    x_max = float(data_frame["tsne_x"].max())
    y_min = float(data_frame["tsne_y"].min())
    y_max = float(data_frame["tsne_y"].max())
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scale = min(plot_width / x_range, plot_height / y_range)

    x_offset = left + (plot_width - (x_range * scale)) / 2.0
    y_offset = bottom + (plot_height - (y_range * scale)) / 2.0

    color_map = build_export_color_map(data_frame, color_field)
    export_df = data_frame.copy()
    export_df["_export_color_value"] = export_df[color_field].fillna("(missing)").astype(str)

    commands = ["q", "/GS1 gs"]
    for bucket, bucket_df in export_df.groupby("_export_color_value", sort=True):
        red, green, blue = hex_to_rgb(color_map.get(bucket, "#4c78a8"))
        commands.append(f"{pdf_number(red)} {pdf_number(green)} {pdf_number(blue)} rg")
        for row in bucket_df.itertuples(index=False):
            screen_x = x_offset + ((float(row.tsne_x) - x_min) * scale)
            screen_y = y_offset + ((float(row.tsne_y) - y_min) * scale)
            commands.extend(circle_path(screen_x, screen_y, EXPORT_POINT_RADIUS))
            commands.append("f")
    commands.append("Q")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    correction_slug = re.sub(r"[^a-z0-9]+", "_", build_info["correction_name"].lower()).strip("_")
    color_slug = re.sub(r"[^a-z0-9]+", "_", color_field.lower()).strip("_")
    output_path = EXPORT_DIR / f"{correction_slug}_{color_slug}_{timestamp}.pdf"
    write_vector_pdf(output_path, width, height, "\n".join(commands))
    return output_path


def render_selected_images(selected_df: pd.DataFrame) -> None:
    if selected_df.empty:
        st.info("Select points in the embedding view or table to show the original images and SEEK codes here.")
        return

    if len(selected_df) == 1:
        row = selected_df.iloc[0]
        st.image(row["original_image_path"], caption="Original image", width="stretch")
        st.code(
            "\n".join(
                [
                    f"atlas_row_id: {row['atlas_row_id']}",
                    f"ele_color_bucket: {row['ele_id']}",
                    f"compact_elephant_id: {row['compact_elephant_id']}",
                    f"original_ele_id: {row['original_ele_id']}",
                    f"label: {row['label']} (original {row['original_label']})",
                    f"subject_id: {row['subject_id']}",
                    f"encounter_id: {row['encounter_id']}",
                    f"ele-SEEK: {row['ele_seek']}",
                    f"subject-SEEK: {row['subject_seek']}",
                ]
            )
        )
        return

    if len(selected_df) > MAX_IMAGES_TO_SHOW:
        st.caption(f"Showing the first {MAX_IMAGES_TO_SHOW} selected points out of {len(selected_df)}.")

    grid_columns = st.columns(3)
    for idx, row in enumerate(selected_df.head(MAX_IMAGES_TO_SHOW).itertuples(index=False)):
        column = grid_columns[idx % len(grid_columns)]
        caption = (
            f"{row.atlas_row_id} | color={row.ele_id} | elephant={row.compact_elephant_id}\n"
            f"ele-SEEK={row.ele_seek}\n"
            f"subject-SEEK={row.subject_seek}"
        )
        with column:
            if row.original_image_exists:
                st.image(row.original_image_path, caption=caption, width="stretch")
            else:
                st.warning(f"Image missing: {row.original_image_path}")


def main() -> None:
    st.set_page_config(
        page_title="Mara Embedding Atlas",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Mara Embedding Atlas")
    st.caption("Switch between the saved correction embeddings and inspect the original elephant image on selection.")

    correction_lookup = {label: code for code, label in CORRECTION_OPTIONS}
    color_lookup = {label: field for field, label in COLOR_OPTIONS}
    default_label = "100% correction"
    default_color_label = "Elephant Bucket"

    with st.sidebar:
        selected_label = st.selectbox("Embedding", [label for _, label in CORRECTION_OPTIONS], index=[label for _, label in CORRECTION_OPTIONS].index(default_label))
        correction_code = correction_lookup[selected_label]
        selected_color_label = st.selectbox("Color by", [label for _, label in COLOR_OPTIONS], index=[label for _, label in COLOR_OPTIONS].index(default_color_label))
        color_field = color_lookup[selected_color_label]
        data_path = THIS_DIR / "data" / f"mara_embedding_{correction_code}_top50.parquet"
        info_path = THIS_DIR / "data" / f"mara_embedding_{correction_code}_top50.build.json"
        if not data_path.exists() or not info_path.exists():
            st.error(
                "Viewer data is missing for this correction. Run "
                "`figures/embedding_atlas_viewer/run_viewer.sh --rebuild` from the repo root."
            )
            st.stop()

    atlas_df, build_info = load_data(correction_code)

    export_column, info_column = st.columns([1, 4])
    with export_column:
        if st.button("Save visualization as PDF", width="stretch"):
            output_path = export_embedding_pdf(atlas_df, build_info, color_field)
            st.success(f"Saved to {output_path}")
    with info_column:
        st.caption(
            "PDF exports are written to `figures/embedding_atlas/exported_visualization` "
            "as vector PDFs using the current `Color by` field, with a clean fit-all view, larger points, and no text or legend."
        )

    with st.sidebar:
        st.subheader("Dataset")
        st.write(f"Correction: {build_info['correction_name']}")
        st.write(f"Rows: {build_info['rows']}")
        st.write(f"Unique labels: {build_info['unique_labels']}")
        st.write(
            "Projection: "
            f"{build_info['correction_name']}, perplexity=10, "
            f"n_iter=1200, seed={build_info['projection_recipe']['projection_seed']}"
        )
        st.write("Top-50 elephant labels are chosen from the 25% correction labels.")
        st.write(f"Current color field: `{color_field}`")
        st.write("You can color by parsed subject- or elephant-level SEEK attributes such as sex, age, tusks, ear marks, and extremes.")
        st.write("Use `compact_elephant_id` and `original_ele_id` below for the true elephant identity.")
        st.write("Select points in the atlas to inspect the original full-size image below.")

    state = embedding_atlas(
        atlas_df,
        key=f"mara_embedding_{correction_code}_{color_field}",
        row_id="atlas_row_id",
        x="tsne_x",
        y="tsne_y",
        text="atlas_text",
        labels="automatic",
        point_size=2.0,
        show_table=False,
        show_charts=False,
        show_embedding=True,
        initial_state=build_initial_state(atlas_df, color_field),
    )

    predicate = state.get("predicate")
    selected_df = query_selection(atlas_df, predicate)

    with st.expander("Selected Images", expanded=not selected_df.empty):
        render_selected_images(selected_df)

    with st.expander("Selected Metadata", expanded=False):
        metadata_columns = [
            "atlas_row_id",
            "pt_row_index",
            "label",
            "ele_id",
            "compact_elephant_id",
            "original_label",
            "original_ele_id",
            "subject_id",
            "encounter_id",
            "ele_seek",
            "subject_seek",
            "original_image_path",
        ]
        if color_field not in metadata_columns:
            metadata_columns.insert(4, color_field)
        st.dataframe(selected_df[metadata_columns], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
