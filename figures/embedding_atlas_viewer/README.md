# Mara Embedding Atlas Viewer

This viewer wraps Apple Embedding Atlas in a small Streamlit app for the `100%` correction embedding.

What it does:
- rebuilds the saved correction projections using the same t-SNE recipe as `figures/tSNE/sweeps/tsne_grid_top50_perp10_ms20_a0.85.pdf`
- uses the top-50 elephant labels selected from the `25%` labels, just like `figures/tSNE/plot.py`
- lets you switch between `0%`, `25%`, `50%`, `75%`, `100%`, and `oracle`
- shows the original Mara image files after you select points in the embedding
- shows both `ele-SEEK` and `subject-SEEK` for the selected points
- reindexes the visible top-50 elephant ids to `E00..E49` so categorical coloring behaves better in Atlas

## Access

### On the cluster node

From the repo root:

```bash
figures/embedding_atlas_viewer/run_viewer.sh
```

This starts Streamlit on the cluster node itself. The launcher prints an SSH tunnel command when it starts.

### From your laptop

If the app is running on a remote cluster node, do not open the node's `127.0.0.1` directly in your laptop browser. Instead, create an SSH tunnel from your laptop:

```bash
ssh -N -L 8501:127.0.0.1:8501 <username>@beery-l40s-3.csail.mit.edu
```

Then open on your laptop:

```text
http://127.0.0.1:8501
```

## Rebuild the cached data

```bash
figures/embedding_atlas_viewer/run_viewer.sh --rebuild
```

## Use a different port

```bash
figures/embedding_atlas_viewer/run_viewer.sh --port=8502
```

If you change the port, use the same port number in your SSH tunnel:

```bash
ssh -N -L 8502:127.0.0.1:8502 <username>@beery-l40s-3.csail.mit.edu
```

## Notes

- The app uses a cached parquet file in `figures/embedding_atlas_viewer/data/`.
- Image display is driven by selection, not hover. Select points in the embedding or table and the original images appear below the Atlas view.
- Because the saved `.pt` files do not include row ids, the metadata join uses the agreed assumption: within each elephant label, points are matched to images in logical dataset order.
