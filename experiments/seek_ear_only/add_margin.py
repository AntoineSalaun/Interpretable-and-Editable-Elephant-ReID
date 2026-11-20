#!/usr/bin/env python3
"""Add the frozen iterative margin policy at budget 8 from verified saved traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.seek_ear_only.run import paired_bootstrap


def add_margin(out):
    source = ROOT / 'experiments/seek_intervention_top20/results'
    rescaled = ROOT / 'experiments/seek_margin_rescaled/results'
    config = json.loads((out / 'configuration.json').read_text())
    assert config['correction_sources'] == ['sighting']
    assert json.loads((out / 'endpoint_audit.json').read_text())['passed']
    previous = json.loads((source / 'prioritization_results.json').read_text())
    for key in ['dataset', 'subset', 'gallery_correction_fn', 'heatmap_reference', 'train_indices_path', 'test_indices_path']:
        assert config['reference_config'][key] == previous['config'][key], key
    meta = np.load(out / 'query_metadata.npz')
    for directory in [source, rescaled]:
        reference_meta = np.load(directory / 'query_metadata.npz')
        for key in ['indices', 'labels', 'encounter_ids']:
            np.testing.assert_array_equal(meta[key], reference_meta[key])
    trace = np.load(source / 'trace_margin_42.npz')['hits']
    np.testing.assert_array_equal(trace, np.load(rescaled / 'trace_margin_42.npz')['hits'])
    audit = json.loads((rescaled / 'rescaling_audit.json').read_text())
    assert audit['raw_vs_original_choice_differences'] == 0
    saved = np.load(out / 'image_hits_sighting.npz')
    old_names = saved['conditions'].tolist()
    base_hits = {name: saved['hits'][:, old_names.index(name)] for name in ['none', 'ear', 'full']}
    np.testing.assert_array_equal(trace[:, 0], base_hits['none'])
    np.testing.assert_array_equal(trace[:, 16], base_hits['full'])
    conditions = ['none', 'ear', 'margin8', 'full']
    hits = np.column_stack([base_hits['none'], base_hits['ear'], trace[:, 8], base_hits['full']])
    per_image = pd.DataFrame(hits, columns=conditions)
    per_image['sighting'] = meta['encounter_ids']
    grouped = per_image.groupby('sighting', sort=False)[conditions]
    scores = grouped.mean()
    counts = grouped.sum()
    per_sighting = pd.read_csv(out / 'per_sighting_sighting.csv', dtype={'sighting': str}).set_index('sighting')
    np.testing.assert_array_equal(scores.index, per_sighting.index)
    decisions = pd.read_csv(rescaled / 'margin_decisions.csv', dtype={'sighting': str})
    first_eight = decisions[decisions.budget <= 8].sort_values(['sighting', 'budget'])
    assert set(first_eight.sighting) == set(scores.index)
    for _, selected in first_eight.groupby('sighting'):
        assert selected.budget.tolist() == list(range(1, 9))
        assert selected.attribute_name.nunique() == 8
    for name in conditions:
        per_sighting[f'recall_at_1_{name}'] = scores[name]
        per_sighting[f'hits_{name}'] = counts[name]
    means = scores.to_numpy().mean(0) * 100
    boot = paired_bootstrap(scores.to_numpy(), config['bootstrap_draws'], config['bootstrap_seed'])
    rows, differences = [], []
    for i, (name, budget) in enumerate(zip(conditions, [0, 8, 8, 16])):
        lo, hi = np.percentile(boot[:, i] - boot[:, 0], [2.5, 97.5])
        rows.append(dict(correction_source='sighting', condition=name, attributes_reviewed=budget,
                         recall_at_1_pct=means[i], improvement_pp=means[i]-means[0], ci95_low_pp=lo, ci95_high_pp=hi))
    for a, b in [(1, 0), (3, 0), (1, 3), (2, 0), (2, 1), (2, 3)]:
        lo, hi = np.percentile(boot[:, a]-boot[:, b], [2.5, 97.5])
        differences.append(dict(correction_source='sighting', contrast=f'{conditions[a]} - {conditions[b]}',
                                difference_pp=means[a]-means[b], ci95_low_pp=lo, ci95_high_pp=hi))
    old_summary = json.loads((out / 'summary.json').read_text())
    # The existing three rows must be unchanged, including their paired intervals.
    for row in old_summary['rows']:
        if row['condition'] in ['none', 'ear', 'full']:
            current = next(r for r in rows if r['condition'] == row['condition'])
            for key in ['recall_at_1_pct', 'improvement_pp', 'ci95_low_pp', 'ci95_high_pp']:
                np.testing.assert_allclose(current[key], row[key], rtol=0, atol=1e-10)
    denominator = means[3]-means[0]
    recovered = lambda i: float(100*(means[i]-means[0])/denominator) if denominator > 0 else None
    recovery = {'sighting': dict(fraction_of_full_improvement_pct=recovered(1),
                                margin8_fraction_of_full_improvement_pct=recovered(2),
                                undefined_reason=None if denominator > 0 else 'Full correction does not improve over no correction.')}
    config['margin8'] = dict(budget=8, selection_unit='sighting; selected attribute applied to every image',
                            recompute_scores_each_round=True, expert_source='ele-SEEK',
                            margin_lambda=previous['margin_lambda']['margin_lambda'],
                            temperature=previous['temperature']['temperature'], equivalent_raw_gap_beta=audit['margin_beta'],
                            trace_source=str(source / 'trace_margin_42.npz'), inference='reuse verified saved iterative trajectory; no retuning')
    inputs = [source / 'trace_margin_42.npz', source / 'prioritization_results.json', source / 'query_metadata.npz',
              rescaled / 'trace_margin_42.npz', rescaled / 'margin_decisions.csv', rescaled / 'rescaling_audit.json']
    margin_audit = dict(query_order_exact_match=True, endpoints_exact_match=True, all_original_and_rescaled_hits_identical=True,
                        eight_distinct_cumulative_choices_per_sighting=True, original_three_rows_unchanged=True,
                        n_sightings=len(scores), n_query_images=len(hits),
                        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    for filename, obj in [('configuration.json', config), ('margin8_audit.json', margin_audit),
                          ('summary.json', dict(rows=rows, paired_differences=differences, recovery=recovery))]:
        (out / filename).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(differences).to_csv(out / 'paired_differences.csv', index=False)
    per_sighting.to_csv(out / 'per_sighting_sighting.csv')
    first_eight.to_csv(out / 'margin8_decisions.csv', index=False)
    np.savez_compressed(out / 'image_hits_sighting.npz', hits=hits, conditions=conditions)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(differences).to_string(index=False))
    print(json.dumps(recovery))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).parent / 'results')
    add_margin(parser.parse_args().results_dir.resolve())
