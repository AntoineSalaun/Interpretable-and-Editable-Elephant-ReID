#!/usr/bin/env python3
"""Reaggregate the fixed comparison as per-image top-1 retrieval accuracy."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).parent / 'results'
    out = source / 'image_weighted'
    out.mkdir(exist_ok=True)
    cfg = json.loads((source / 'configuration.json').read_text())
    assert json.loads((source / 'validation.json').read_text())['passed']
    data = np.load(source / 'image_hits_sighting.npz')
    old_conditions, saved_hits = data['conditions'].tolist(), data['hits']
    query = np.load(source / 'query_metadata.npz')
    prior = root / 'experiments/seek_intervention_top20/results'
    prior_meta = np.load(prior / 'query_metadata.npz')
    for key in ['indices', 'labels', 'encounter_ids']:
        np.testing.assert_array_equal(query[key], prior_meta[key])
    seeds = list(range(42, 62))
    random_paths = [prior / f'trace_random_{seed}.npz' for seed in seeds]
    random_hits = []
    for path in random_paths:
        trace = np.load(path)['hits']
        np.testing.assert_array_equal(trace[:, 0], saved_hits[:, old_conditions.index('none')])
        np.testing.assert_array_equal(trace[:, 16], saved_hits[:, old_conditions.index('full')])
        random_hits.append(trace[:, 8])
    random_hits = np.stack(random_hits, axis=1)
    conditions = ['random8', 'ear', 'margin8', 'full']
    hits = np.column_stack([random_hits.mean(1), *[saved_hits[:, old_conditions.index(c)] for c in conditions[1:]]])
    frame = pd.DataFrame(hits, columns=conditions)
    frame['sighting'] = query['encounter_ids']
    grouped = frame.groupby('sighting', sort=False)
    totals, sizes = grouped[conditions].sum().to_numpy(), grouped.size().to_numpy()
    random_frame = pd.DataFrame(random_hits, columns=seeds)
    random_frame['sighting'] = query['encounter_ids']
    random_totals = random_frame.groupby('sighting', sort=False)[seeds].sum().to_numpy()
    means = hits.mean(axis=0) * 100
    np.testing.assert_allclose(means, totals.sum(0)/sizes.sum()*100, rtol=0, atol=1e-12)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    seed_rng = np.random.default_rng(cfg['bootstrap_seed']+1)
    boot = np.empty((cfg['bootstrap_draws'], len(conditions)))
    for start in range(0, len(boot), 100):
        sampled = rng.integers(0, len(sizes), size=(min(100, len(boot)-start), len(sizes)))
        # Shared sighting resamples; total image hits / total images in each sample.
        boot[start:start+len(sampled)] = totals[sampled].sum(1)/sizes[sampled].sum(1)[:, None]*100
        # Also resample the 20 independent random-policy runs, keeping each run
        # intact across the shared sighting sample (crossed seed/sighting bootstrap).
        sampled_seeds = seed_rng.integers(0, len(seeds), size=(len(sampled), len(seeds)))
        random_sum = random_totals[sampled].sum(1)
        boot[start:start+len(sampled), 0] = np.take_along_axis(random_sum, sampled_seeds, axis=1).mean(1)/sizes[sampled].sum(1)*100
    baseline, full = conditions.index('random8'), conditions.index('full')
    rows, contrasts = [], []
    budgets = {'random8': 8, 'ear': 8, 'margin8': 8, 'full': 16}
    denominator = means[full]-means[baseline]
    for i, name in enumerate(conditions):
        lo, hi = np.percentile(boot[:, i]-boot[:, baseline], [2.5, 97.5])
        rows.append(dict(condition=name, attributes_reviewed=budgets[name], accuracy_pct=means[i],
                         improvement_pp=means[i]-means[baseline], ci95_low_pp=lo, ci95_high_pp=hi,
                         recovered_gain_pct=float(100*(means[i]-means[baseline])/denominator) if denominator > 0 else None))
    for a, b in [('ear', 'random8'), ('margin8', 'random8'), ('full', 'random8'),
                 ('margin8', 'ear'), ('ear', 'full'), ('margin8', 'full')]:
        i, j = conditions.index(a), conditions.index(b)
        lo, hi = np.percentile(boot[:, i]-boot[:, j], [2.5, 97.5])
        contrasts.append(dict(contrast=f'{a} - {b}', difference_pp=means[i]-means[j], ci95_low_pp=lo, ci95_high_pp=hi))
    # Reproduce original endpoints using their saved hit vectors, in float64 here.
    original = np.load(root / 'experiments/seek_intervention_top20/results/trace_margin_42.npz')['hits']
    np.testing.assert_array_equal(saved_hits[:, old_conditions.index('none')], original[:, 0])
    np.testing.assert_array_equal(hits[:, full], original[:, 16])
    np.testing.assert_array_equal(hits[:, conditions.index('margin8')], original[:, 8])
    cfg.update(metric='top-1 retrieval accuracy (image-weighted Recall@1)',
               baseline='eight uniform random distinct attribute corrections, mean over 20 runs',
               random_policy_seeds=seeds, random_budget=8, seed_bootstrap_rng=cfg['bootstrap_seed']+1,
               random_selection='without replacement; one choice per sighting per round, applied to every image',
               recovered_gain_definition='100 * (accuracy_condition - accuracy_random8) / (accuracy_full - accuracy_random8)',
               aggregation='sum of correct top-1 image queries / total query images',
               bootstrap_estimator='paired sighting resampling; additionally resample the 20 random-policy runs with replacement; total hits / total images',
               supersedes_metric='equal mean of per-sighting recalls; previous outputs retained in parent directory')
    cfg['input_sha256'] = {str(source / p): hashlib.sha256((source / p).read_bytes()).hexdigest()
                           for p in ['image_hits_sighting.npz', 'query_metadata.npz', 'configuration.json']}
    cfg['input_sha256'].update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in random_paths})
    result = dict(rows=rows, paired_differences=contrasts, endpoints_exact_match=True,
                  recovery_undefined_reason=None if denominator > 0 else 'Full correction does not improve over eight random corrections.')
    for filename, value in [('configuration.json', cfg), ('summary.json', result)]:
        (out / filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'paired_differences.csv', index=False)
    pd.DataFrame({'seed': seeds, 'accuracy_pct': random_hits.mean(0)*100}).to_csv(out / 'random8_per_seed.csv', index=False)
    pd.DataFrame(hits, columns=conditions).assign(image_index=query['indices'], sighting=query['encounter_ids']).to_csv(out / 'per_image_results.csv', index=False)
    grouped.mean().assign(n_images=sizes).to_csv(out / 'per_sighting_results.csv')
    table = {r['condition']: r for r in rows}
    paragraph = (
        r'Motivated by the importance of ear morphology, we evaluated a simple heuristic on the same held-out queries '
        r'and fixed $G_S(20\%)$ gallery, comparing eight randomly selected corrections, correction of only the eight ear-hole and tear '
        r'attributes, eight iterative margin-ratio corrections, and correction of all 16 SEEK attributes. '
        r'Corrections used expert sighting-level values applied to each image, while all remaining attributes '
        r'retained their model predictions. '
        f"Ear-only correction achieved {table['ear']['accuracy_pct']:.1f}\\% top-1 retrieval accuracy, compared with "
        f"{table['random8']['accuracy_pct']:.1f}\\% with eight random corrections (averaged over 20 runs), {table['margin8']['accuracy_pct']:.1f}\\% with "
        f"eight margin-ratio corrections, and {table['full']['accuracy_pct']:.1f}\\% with full correction"
    )
    if denominator > 0:
        paragraph += (f", recovering {table['ear']['recovered_gain_pct']:.1f}\\% of the improvement from eight random corrections to full correction "
                      'while requiring review of half as many attributes.')
    else:
        paragraph += '; the fraction of improvement recovered was undefined because full correction did not improve over eight random corrections.'
    (out / 'section_4_2.tex').write_text(paragraph+'\n')
    text = ['# Top-1 retrieval accuracy: each image has equal weight', '',
            'These are the current requested results. The parent directory retains the earlier sighting-averaged analysis. '
            'The model, gallery, annotation source and queries are unchanged. The baseline is now eight random '
            'corrections, averaged over all 20 previously saved policy seeds (42--61). Gains and recovered fractions '
            'are relative to this baseline, not no correction.', '',
            '| Condition | Attributes reviewed | Accuracy (%) | Gain (pp) | Paired 95% CI (pp) | Gain recovered (%) |',
            '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        recovered = f"{r['recovered_gain_pct']:.4f}" if r['recovered_gain_pct'] is not None else 'undefined'
        text.append(f"| {r['condition']} | {r['attributes_reviewed']} | {r['accuracy_pct']:.4f} | "
                    f"{r['improvement_pp']:.4f} | [{r['ci95_low_pp']:.4f}, {r['ci95_high_pp']:.4f}] | {recovered} |")
    text += ['', 'Accuracy is the percentage of query images whose highest-ranked gallery image has the correct identity. '
             'The 10,000 paired bootstrap replicates resample sightings (seed 2026) and independently resample the '
             '20 random-policy runs (seed 2027), then divide total correct images by total images in each resample. '
             'Sampled sightings are shared across all conditions, and sampled random runs are shared across all '
             'contrasts within a replicate. Per-image random results are mean hit rates across runs. '
             'Reviewed attributes are counts, not measured annotation time. '
             'Original endpoint image-hit vectors and margin budget-8 hits match exactly; endpoint percentages are '
             'computed in float64 here, versus the original evaluator\'s float32 mean.', '', paragraph]
    (out / 'report.md').write_text('\n'.join(text)+'\n')
    (out / 'reproduce.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(root))+'\n'+
                                    shlex.join([sys.executable, str(Path(__file__).resolve())])+'\n')
    sys.path.insert(0, str(root))
    from experiments.seek_ear_only.report import render_current
    render_current(source)
    print(pd.DataFrame(rows).to_string(index=False))
    print(paragraph)


if __name__ == '__main__':
    main()
