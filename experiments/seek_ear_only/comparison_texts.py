#!/usr/bin/env python3
"""Two image-accuracy comparisons: all strategies at eight, and 0/8/16 corrections."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

LABELS = dict(random8='Random (8)', oracle8='Greedy oracle (8)', none='No correction',
              ear='Ear-only (8)', margin8='Margin-ratio (8)', full='Full correction (16)')
GROUPS = dict(budget8=['random8', 'oracle8', 'ear', 'margin8'],
              correction_extent=['none', 'full', 'ear', 'margin8'])


def render(out):
    current = out / 'comparisons'
    result = json.loads((current / 'summary.json').read_text())
    rows = {r['condition']: r for r in result['rows']}
    lines = ['# Section 4.2: two correction comparisons', '',
             'Accuracy is top-1 retrieval accuracy with equal weight for every query image. '
             'All conditions share the unchanged checkpoint, fixed G_S(20%) gallery, and 16,506 query images '
             'from 1,269 sightings. Corrections use expert sighting-level `ele-SEEK` values applied per image, '
             'including unknowns; remaining attributes retain their predictions.', '']
    for key, names in GROUPS.items():
        lines += ['## '+('Eight corrections for every strategy' if key == 'budget8' else 'No, partial, and full correction'), '',
                  '| Condition | Attributes reviewed per image | Accuracy (%) |', '|---|---:|---:|']
        for name in names:
            r = rows[name]
            lines.append(f"| {LABELS[name]} | {r['attributes_reviewed']} | {r['accuracy_pct']:.1f} |")
        lines += ['', '```latex', (current / f'{key}.tex').read_text().strip(), '```', '']
    lines += ['## Paired uncertainty', '',
              '| Comparison | Difference (pp) | Paired 95% CI (pp) |', '|---|---:|---:|']
    for r in result['paired_differences']:
        a, b = r['contrast'].split(' - ')
        lines.append(f"| {LABELS[a]} − {LABELS[b]} | {r['difference_pp']:+.1f} | [{r['ci95_low_pp']:.1f}, {r['ci95_high_pp']:.1f}] |")
    lines += ['', 'Ear-only is numerically above margin-ratio; their paired interval includes zero, '
              'so this comparison does not establish a statistically significant difference.', '',
              'Intervals use 10,000 paired sighting resamples (seed 2026), computing total correct images '
              'divided by total images in each sample. Random averages all 20 original runs (seeds 42–61); '
              'those runs are additionally resampled with replacement using independent RNG seed 2027. '
              'Shared sighting samples preserve pairing across conditions. These intervals are conditional '
              'on the fixed model and gallery; different sightings of one identity are not clustered.', '',
              'Random, margin-ratio, and greedy oracle choose an attribute per sighting each round and apply '
              'it to every image, giving eight distinct corrected attributes per image at budget eight. '
              'Margin-ratio and greedy oracle recompute scores after every correction. Greedy oracle uses '
              'expert values and the true identity to select the correction with the best immediate retrieval '
              'result; it is a truth-aware reference. Attribute counts are not measured annotation time.', '',
              'The same no/full endpoint hit vectors and query ordering are verified for all 20 random '
              'traces and the oracle trace. Oracle and margin traces match the independently verified '
              'raw-margin run. No retraining, inference rerun, checkpoint selection, or retuning was needed.', '',
              '## Reproduction', '', '```bash', 'python experiments/seek_ear_only/comparison_texts.py', '```', '',
              'Current full-precision tables, paired differences, per-image outcomes, configuration, hashes '
              'and both standalone LaTeX paragraphs are in `results/comparisons/`. Earlier comparisons '
              'remain in `results/image_weighted/`; the historical sighting-averaged outputs remain in the parent directory.']
    text = '\n'.join(lines)+'\n'
    (current / 'report.md').write_text(text)
    (out / 'report.md').write_text(text)
    (out / 'section_4_2.tex').write_text('\n\n'.join((current / f'{k}.tex').read_text().strip() for k in GROUPS)+'\n')


def main():
    root = Path(__file__).resolve().parents[2]
    out = Path(__file__).parent / 'results'
    target = out / 'comparisons'
    target.mkdir(exist_ok=True)
    cfg = json.loads((out / 'configuration.json').read_text())
    assert json.loads((out / 'endpoint_audit.json').read_text())['passed']
    saved = np.load(out / 'image_hits_sighting.npz')
    old = {c: saved['hits'][:, i] for i, c in enumerate(saved['conditions'])}
    meta = np.load(out / 'query_metadata.npz')
    prior = root / 'experiments/seek_intervention_top20/results'
    rescaled = root / 'experiments/seek_margin_rescaled/results'
    inputs = [out / 'image_hits_sighting.npz', out / 'query_metadata.npz', out / 'configuration.json']
    for directory in [prior, rescaled]:
        other = np.load(directory / 'query_metadata.npz')
        for key in ['indices', 'encounter_ids', 'labels']:
            np.testing.assert_array_equal(meta[key], other[key])
    def trace(name):
        path = prior / name
        inputs.append(path)
        value = np.load(path)['hits']
        np.testing.assert_array_equal(value[:, 0], old['none'])
        np.testing.assert_array_equal(value[:, 16], old['full'])
        return value
    oracle = trace('trace_oracle_42.npz')
    for name, values in [('trace_oracle_42.npz', oracle), ('trace_margin_42.npz', trace('trace_margin_42.npz'))]:
        np.testing.assert_array_equal(values, np.load(rescaled / name)['hits'])
        inputs.append(rescaled / name)
    random = np.stack([trace(f'trace_random_{s}.npz')[:, 8] for s in range(42, 62)], axis=1)
    conditions = list(LABELS)
    hits = np.column_stack([random.mean(1), oracle[:, 8], old['none'], old['ear'], old['margin8'], old['full']])
    frame = pd.DataFrame(hits, columns=conditions).assign(sighting=meta['encounter_ids'])
    grouped = frame.groupby('sighting', sort=False)
    totals, sizes = grouped[conditions].sum().to_numpy(), grouped.size().to_numpy()
    rt = pd.DataFrame(random).assign(sighting=meta['encounter_ids']).groupby('sighting', sort=False).sum().to_numpy()
    rng, srng = np.random.default_rng(2026), np.random.default_rng(2027)
    boot = np.empty((10000, len(conditions)))
    for start in range(0, len(boot), 100):
        sample = rng.integers(0, len(sizes), (100, len(sizes)))
        denom = sizes[sample].sum(1)
        boot[start:start+100] = totals[sample].sum(1)/denom[:, None]*100
        seed_sample = srng.integers(0, 20, (100, 20))
        boot[start:start+100, 0] = np.take_along_axis(rt[sample].sum(1), seed_sample, axis=1).mean(1)/denom*100
    means = hits.mean(0)*100
    rows = [dict(condition=c, attributes_reviewed=0 if c == 'none' else 16 if c == 'full' else 8,
                 accuracy_pct=means[i]) for i, c in enumerate(conditions)]
    contrasts = []
    for a, b in [('ear', 'margin8'), ('oracle8', 'ear'), ('oracle8', 'margin8'), ('oracle8', 'random8'),
                 ('ear', 'random8'), ('margin8', 'random8'), ('ear', 'none'), ('margin8', 'none'),
                 ('full', 'none'), ('ear', 'full'), ('margin8', 'full')]:
        i, j = conditions.index(a), conditions.index(b)
        lo, hi = np.percentile(boot[:, i]-boot[:, j], [2.5, 97.5])
        contrasts.append(dict(contrast=f'{a} - {b}', difference_pp=means[i]-means[j], ci95_low_pp=lo, ci95_high_pp=hi))
    values = dict(zip(conditions, means))
    denominator = values['full']-values['none']
    recovered = float(100*(values['ear']-values['none'])/denominator) if denominator > 0 else None
    pair = contrasts[0]
    assert pair['ci95_low_pp'] < 0 < pair['ci95_high_pp'], 'Revise nonsignificance wording if this changes'
    first = (
        r'Motivated by the importance of ear morphology, we evaluate a simple heuristic that always corrects '
        r'the same eight ear-hole and tear attributes, using the fixed $G_S(20\%)$ gallery and a budget of '
        r'eight corrections per query. '
        f"At this budget, the \\textit{{random}} strategy achieves {values['random8']:.1f}\\% top-1 retrieval "
        f"accuracy (averaged over 20 runs) and the greedy oracle achieves {values['oracle8']:.1f}\\% "
        r'(Fig.~\ref{fig:seek-prioritization}). '
        f"The ear-correction heuristic achieves {values['ear']:.1f}\\%, slightly above the margin-ratio "
        f"prioritisation strategy ({values['margin8']:.1f}\\%), although the paired difference is not statistically significant."
    )
    second = (
        r'Under the same fixed $G_S(20\%)$ gallery, we compare no correction, full correction of all 16 SEEK '
        r'attributes, the eight-attribute ear-correction heuristic, and margin-ratio prioritisation with a '
        r'budget of eight corrections per query. '
        f"Top-1 retrieval accuracy is {values['none']:.1f}\\% without correction and {values['full']:.1f}\\% "
        f"with full correction, compared with {values['ear']:.1f}\\% for ear-only correction and "
        f"{values['margin8']:.1f}\\% for margin-ratio prioritisation. "
        + (f"Ear-only correction recovers {recovered:.1f}\\% of the full-correction improvement over no correction "
           'while requiring review of half as many attributes; its paired difference from margin-ratio is not statistically significant.'
           if recovered is not None else 'The recovered fraction is undefined because the full-correction gain is nonpositive.')
    )
    cfg.update(metric='top-1 retrieval accuracy; equal image weights', comparisons=GROUPS,
               bootstrap_draws=10000, bootstrap_seed=2026, random_seed_bootstrap=2027,
               random_policy_seeds=list(range(42,62)), oracle_budget=8, oracle_truth_aware=True,
               aggregation='total correct image queries divided by total image queries',
               bootstrap_estimator='shared sighting resamples; additionally resample random-policy runs',
               input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    result = dict(rows=rows, paired_differences=contrasts, ear_recovered_full_gain_over_none_pct=recovered,
                  oracle_endpoints_exact_match=True, oracle_trace_matches_verified_rerun=True, query_order_exact_match=True)
    for name, obj in [('summary.json', result), ('configuration.json', cfg)]:
        (target / name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    pd.DataFrame(rows).to_csv(target / 'summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(target / 'paired_differences.csv', index=False)
    frame.assign(image_index=meta['indices']).to_csv(target / 'per_image_results.csv', index=False)
    for key, names in GROUPS.items():
        pd.DataFrame(rows).set_index('condition').loc[names].to_csv(target / f'{key}.csv')
    for name, text in [('budget8.tex', first), ('correction_extent.tex', second)]:
        (target / name).write_text(text+'\n')
    (target / 'reproduce.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(root))+'\n'+
                                      shlex.join([sys.executable,str(Path(__file__).resolve())])+'\n')
    render(out)
    print(first+'\n\n'+second)
    print(json.dumps(pair))


if __name__ == '__main__':
    main()
