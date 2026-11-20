#!/usr/bin/env python3
"""Verify saved aggregation, split integrity, and endpoints against both prior runs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    out = Path(__file__).parent / 'results'
    cfg = json.loads((out / 'configuration.json').read_text())
    summary = json.loads((out / 'summary.json').read_text())
    query = np.load(out / 'query_metadata.npz')
    gallery = np.load(out / 'gallery_metadata.npz')
    assert not set(query['indices']) & set(gallery['indices'])
    assert not set(query['encounter_ids']) & set(gallery['encounter_ids'])
    expected_train = np.loadtxt(cfg['resolved_train_indices'], dtype=int)
    expected_test = np.loadtxt(cfg['resolved_test_indices'], dtype=int)
    np.testing.assert_array_equal(gallery['indices'], expected_train)
    np.testing.assert_array_equal(np.sort(query['indices']), np.sort(expected_test))
    image_results = np.load(out / 'image_hits_sighting.npz')
    hits = image_results['hits']
    conditions = image_results['conditions'].tolist()
    frame = pd.DataFrame(hits, columns=conditions)
    frame['sighting'] = query['encounter_ids']
    aggregated = frame.groupby('sighting', sort=False)[conditions].mean()
    saved = pd.read_csv(out / 'per_sighting_sighting.csv', dtype={'sighting': str}).set_index('sighting')
    np.testing.assert_array_equal(aggregated.index, saved.index)
    np.testing.assert_allclose(aggregated.to_numpy(), saved[[f'recall_at_1_{c}' for c in conditions]], atol=1e-15)
    for row in summary['rows']:
        np.testing.assert_allclose(row['recall_at_1_pct'], aggregated[row['condition']].mean()*100, atol=1e-12)
    trace_count = 0
    for directory in ['seek_intervention_top20', 'seek_margin_rescaled']:
        source = root / 'experiments' / directory / 'results'
        reference_meta = np.load(source / 'query_metadata.npz')
        for key in ['indices', 'labels', 'encounter_ids']:
            np.testing.assert_array_equal(query[key], reference_meta[key])
        for trace in source.glob('trace_*.npz'):
            old = np.load(trace)['hits']
            np.testing.assert_array_equal(hits[:, [conditions.index('none'), conditions.index('full')]], old[:, [0, 16]])
            if 'margin8' in conditions and trace.name == 'trace_margin_42.npz':
                np.testing.assert_array_equal(hits[:, conditions.index('margin8')], old[:, 8])
            trace_count += 1
    result = dict(passed=True, gallery_indices_exact_match=True, query_indices_exact_match=True,
                  train_test_image_overlap=0, train_test_sighting_overlap=0,
                  per_sighting_results_recomputed=True, n_sightings=len(aggregated),
                  preceding_policy_traces_with_exact_endpoint_match=trace_count,
                  source_runs=['seek_intervention_top20', 'seek_margin_rescaled'],
                  margin8_saved_trajectory_exact_match='margin8' in conditions)
    (out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
