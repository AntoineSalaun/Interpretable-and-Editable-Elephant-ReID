#!/usr/bin/env python3
"""Render the current image-accuracy comparison with random-at-eight baseline."""
import argparse
import json
from pathlib import Path
import sys


def render_current(out):
    if (out / 'comparisons/summary.json').exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from experiments.seek_ear_only.comparison_texts import render
        render(out)
        print('Updated both current comparisons:', out / 'report.md')
        return
    current = out / 'image_weighted'
    summary = json.loads((current / 'summary.json').read_text())
    cfg = json.loads((current / 'configuration.json').read_text())
    assert summary['endpoints_exact_match']
    assert json.loads((out / 'endpoint_audit.json').read_text())['passed']
    rows = {r['condition']: r for r in summary['rows']}
    labels = {'random8': 'Eight random corrections', 'ear': 'Ear-only correction',
              'margin8': 'Eight iterative margin-ratio corrections', 'full': 'Full correction'}
    assert list(rows) == list(labels)
    paragraph = (current / 'section_4_2.tex').read_text().strip()
    lines = ['# Section 4.2: ear-only correction versus random and margin-ratio', '',
             'We compare eight random corrections, the eight ear-hole/tear corrections, eight iterative '
             'margin-ratio corrections, and all 16 SEEK corrections. Accuracy is the percentage of query '
             'images whose top-ranked gallery image has the correct identity (Recall@1); each query image '
             'has equal weight. Random accuracy is averaged over 20 policy runs (seeds 42–61).', '',
             '| Condition | Attributes reviewed per image | Accuracy (%) | Gain over random (pp) | Paired 95% CI for gain (pp) | Improvement recovered (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for name, label in labels.items():
        r = rows[name]
        recovered = 'undefined' if r['recovered_gain_pct'] is None else f"{r['recovered_gain_pct']:.1f}"
        ci = '—' if name == 'random8' else f"[{r['ci95_low_pp']:.1f}, {r['ci95_high_pp']:.1f}]"
        lines.append(f"| {label} | {r['attributes_reviewed']} | {r['accuracy_pct']:.1f} | "
                     f"{r['improvement_pp']:.1f} | {ci} | {recovered} |")
    ear = rows['ear']
    recovery = ('The recovered fraction is undefined because the full-correction gain over random is nonpositive.'
                if ear['recovered_gain_pct'] is None else
                f"Ear-only correction recovers **{ear['recovered_gain_pct']:.1f}%** of the improvement from eight random "
                'corrections to full correction, with half as many reviewed attributes as full correction.')
    lines += ['', recovery, '',
              'The recovered fraction is `100 × (accuracy_ear − accuracy_random8) / '
              '(accuracy_full − accuracy_random8)`. Reviewed-attribute counts are not measured annotation time.', '',
              '## Paired comparisons', '',
              '| Comparison | Accuracy difference (pp) | Paired 95% CI (pp) |', '|---|---:|---:|']
    for r in summary['paired_differences']:
        a, b = r['contrast'].split(' - ')
        lines.append(f"| {labels[a]} − {labels[b]} | {r['difference_pp']:+.1f} | "
                     f"[{r['ci95_low_pp']:.1f}, {r['ci95_high_pp']:.1f}] |")
    contrast = next(r for r in summary['paired_differences'] if r['contrast'] == 'margin8 - ear')
    if contrast['ci95_low_pp'] <= 0 <= contrast['ci95_high_pp']:
        lines += ['', 'The paired interval for margin-ratio versus ear-only includes zero; this comparison '
                  'does not establish a difference between them at the same eight-attribute budget.']
    lines += ['', '## Fixed setup and correction procedure', '',
              f"The unchanged `Heatmap_three_20cor` checkpoint and fixed $G_S(20\\%)$ gallery use "
              f"{cfg['n_gallery_images']:,} gallery images and {cfg['n_query_images']:,} held-out query images "
              f"from {cfg['n_query_sightings']:,} sightings and {cfg['n_query_identities']:,} identities. "
              'Corrections copy expert sighting-level `ele-SEEK` categorical values into each image, including '
              'unknown/occluded values; all remaining attributes keep their hard model predictions. '
              'Correction precedes per-image projection, visual/SEEK fusion, and full-gallery cosine retrieval.', '',
              'The eight ear-hole/tear attributes are selected from the dataset schema: '
              + ', '.join(f'`{name}`' for name in cfg['ear_attributes']) + '. '
              'Random selects eight distinct attributes without replacement. Margin-ratio recomputes scores '
              'after each correction using its previously frozen parameters. Both policies choose attributes '
              'per sighting and apply them to every image in that sighting, giving eight corrections per image. '
              'No retraining, checkpoint selection, or parameter tuning was performed.', '',
              '## Confidence intervals and validation', '',
              'The 10,000 paired percentile bootstrap replicates resample query sightings with replacement '
              '(seed 2026), sharing sampled sightings across conditions. Each replicate uses total correct '
              'images divided by total sampled images. Independently resampling the 20 random-policy runs '
              '(seed 2027) includes finite-run uncertainty; a sampled run is kept intact across the sighting sample. '
              'Intervals are conditional on the fixed checkpoint and gallery and do not cluster different '
              'sightings of the same identity.', '',
              'The original endpoint hit vectors reproduce exactly. Margin-ratio budget-8 hits match both '
              'preceding runs, and all 20 random traces share the verified endpoints and query ordering. '
              'The image-accuracy baseline change leaves the ear-only, margin-ratio, and full-correction '
              'predictions unchanged. Full-precision results and input hashes are in `image_weighted/`.', '',
              '## Section 4.2 LaTeX draft', '', '```latex', paragraph, '```', '',
              '## Reproduction and files', '',
              'Regenerate the current statistics and report from saved predictions:', '', '```bash',
              'python experiments/seek_ear_only/image_weighted.py', '```', '',
              'For a report-only refresh, run `python experiments/seek_ear_only/report.py`. '
              '`results/reproduce.sh` runs the full inference/trace-reuse workflow and then generates this report. '
              'Current configuration, summary tables, per-image/per-sighting results, random results by seed, '
              'paired differences, and the standalone LaTeX paragraph are in `results/image_weighted/`. '
              'The parent directory’s historical numeric tables retain the earlier sighting-averaged analysis.']
    report = '\n'.join(lines)+'\n'
    for directory in [out, current]:
        (directory / 'report.md').write_text(report)
        (directory / 'section_4_2.tex').write_text(paragraph+'\n')
    print('Updated current report and LaTeX:', out / 'report.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path(__file__).parent / 'results')
    p.add_argument('--legacy-sighting-averaged', action='store_true', help='Render the historical sighting-averaged report instead.')
    args = p.parse_args()
    out = args.results_dir
    if not args.legacy_sighting_averaged:
        render_current(out)
        return
    summary = json.loads((out / 'summary.json').read_text())
    cfg = json.loads((out / 'configuration.json').read_text())
    audit = json.loads((out / 'endpoint_audit.json').read_text())
    assert audit['passed'], 'Do not interpret results before endpoint verification'
    assert cfg['correction_sources'] == ['sighting'], 'This report is for the agreed prior correction source'
    rows = {r['condition']: r for r in summary['rows']}
    none, ear, full = [rows[k] for k in ['none', 'ear', 'full']]
    margin = rows.get('margin8')
    ratio = summary['recovery']['sighting']['fraction_of_full_improvement_pct']
    lines = ['# Section 4.2: fixed ear-only heuristic', '',
             'The confirmed correction source is the preceding experiment\'s sighting-level `ele-SEEK`, '
             'copied into each query image. The requested image-level `subject-SEEK` source was superseded '
             'by the user\'s clarification before the reported comparison.', '',
             f"The unchanged checkpoint and fixed G_S(20%) gallery contain {cfg['n_gallery_images']:,} gallery images. "
             f"There are {cfg['n_query_images']:,} held-out query images in {cfg['n_query_sightings']:,} sightings "
             f"from {cfg['n_query_identities']:,} identities. Correct each image's categorical SEEK slices first, "
             'then project/fuse and retrieve per image; average hits within each sighting, then equally over sightings. '
             'Unknown/occluded categories are copied unchanged.', '',
             '| Condition | Attributes reviewed | Sighting-averaged Recall@1 (%) | Gain (pp) | Paired 95% CI for gain (pp) |',
             '|---|---:|---:|---:|---:|']
    display = [('none', 'No correction'), ('ear', 'Ear-only')]
    if margin:
        display.append(('margin8', 'Iterative margin-ratio (8 corrections)'))
    display.append(('full', 'Full correction'))
    for condition, label in display:
        r = rows[condition]
        lines.append(f"| {label} | {r['attributes_reviewed']} | {r['recall_at_1_pct']:.4f} | "
                     f"{r['improvement_pp']:.4f} | [{r['ci95_low_pp']:.4f}, {r['ci95_high_pp']:.4f}] |")
    lines += ['', 'Counts of reviewed attributes are not measured annotation time.', '']
    if ratio is None:
        conclusion = 'The recovered fraction is undefined because full correction does not improve Recall@1 over no correction.'
    else:
        conclusion = (f'Ear-only correction recovers {ratio:.2f}% of the full-correction improvement. '
                      + ('It recovers most of the measured improvement.' if ratio > 50 else
                         'It does not recover most of the measured improvement.'))
    lines += [conclusion, '', 'All paired contrasts:', '']
    for r in summary['paired_differences']:
        lines.append(f"- {r['contrast']}: {r['difference_pp']:.4f} pp "
                     f"(95% CI [{r['ci95_low_pp']:.4f}, {r['ci95_high_pp']:.4f}]).")
    if margin:
        recovered = summary['recovery']['sighting']['margin8_fraction_of_full_improvement_pct']
        lines += ['', 'The added margin-ratio row reuses budget 8 of the saved iterative policy trajectory: '
                  'scores are recomputed after each correction, eight distinct attributes are chosen per sighting, '
                  'and each chosen attribute is corrected in every image of that sighting. '
                  'This gives eight corrected attributes per image; selection is shared across the sighting. '
                  'The original lambda=0.3 and temperature=0.03548133373260498 remain frozen. '
                  'The verified equivalent raw-gap implementation uses beta=8.455150030741938 and matches every '
                  'image hit at every budget. No model inference or tuning was needed to add this saved result.', '',
                  f'Margin-ratio recovers {recovered:.2f}% of the full-correction improvement.' if recovered is not None else
                  'Margin-ratio recovery is undefined because the full-correction gain is nonpositive.']
    lines += ['', f"Intervals use {cfg['bootstrap_draws']:,} percentile bootstrap replicates (seed {cfg['bootstrap_seed']}), "
              'resampling query sightings with replacement and sharing the sampled sightings across conditions. '
              'They are conditional on the fixed model and gallery; repeated sightings of one identity are not clustered.', '',
              '## Endpoint reproduction', '',
              f"Both endpoint image-hit vectors match exactly: {audit['image_hit_mismatches']} mismatches "
              f"across {cfg['n_query_images']:,} images. Original image-weighted Recall@1 is "
              f"{audit['image_weighted_recall_pct'][0]:.8f}% / {audit['image_weighted_recall_pct'][1]:.8f}%; "
              'the numerical tolerance is 1e-6 percentage points. Reaggregating the saved reference image hits '
              'by sighting reproduces both primary endpoint values exactly. The difference from the preceding '
              'main figure\'s percentages is solely its equal-image versus the requested equal-sighting weighting.', '',
              'The original split symlinks and preprocessed-image paths had become stale after an archive move. '
              'The relocated train/test files match the archived originals byte for byte (SHA-256 recorded in '
              '`relocation_audit.json`); image path prefixes were redirected to the archived original crops '
              'without changing suffixes or pixels. No checkpoint, split, correction regime, or retrieval setting '
              'was selected based on results.', '',
              'Run `bash experiments/seek_ear_only/results/reproduce.sh` from any directory to rerun inference '
              '(the validated feature cache may be reused), add the saved margin row, and regenerate this report and LaTeX. '
              'To add only the margin row to existing base results, run '
              '`python experiments/seek_ear_only/add_margin.py` followed by '
              '`python experiments/seek_ear_only/report.py` from the repository root.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    fraction_text = (r'The recovered fraction was undefined because full correction did not improve over no correction'
                     if ratio is None else
                     f"Ear-only correction recovered {ratio:.2f}\\% of the full-correction improvement"
                     + (', hence most of the measured gain' if ratio > 50 else ', less than half of the measured gain'))
    paragraph = (
        r'Using the unchanged checkpoint and fixed $G_S(20\%)$ gallery, we compared no correction, correction of the eight '
        r'ear-hole/tear attributes, and correction of all 16 attributes on '
        f"{cfg['n_query_sightings']:,} held-out query sightings, preserving the preceding experiment's "
        r'\texttt{ele-SEEK} expert values (including unknowns), applying corrections before per-image retrieval, '
        r'and averaging Recall@1 first within and then equally across sightings. '
        f"Recall@1 was {none['recall_at_1_pct']:.2f}\\%, {ear['recall_at_1_pct']:.2f}\\%, and "
        f"{full['recall_at_1_pct']:.2f}\\%, respectively; ear-only and full correction improved over no correction by "
        f"{ear['improvement_pp']:.2f} and {full['improvement_pp']:.2f} percentage points "
        f"(paired 95\\% bootstrap intervals [{ear['ci95_low_pp']:.2f}, {ear['ci95_high_pp']:.2f}] and "
        f"[{full['ci95_low_pp']:.2f}, {full['ci95_high_pp']:.2f}], respectively; "
        f"{cfg['bootstrap_draws']:,} shared sighting resamples). "
        + fraction_text + r'; the respective review budgets were 0, 8, and 16 attributes, not measured annotation times.'
    )
    if margin:
        contrast = next(r for r in summary['paired_differences'] if r['contrast'] == 'margin8 - ear')
        recovered = summary['recovery']['sighting']['margin8_fraction_of_full_improvement_pct']
        recovery_text = (f"Ear-only and margin-ratio recovered {ratio:.2f}\\% and {recovered:.2f}\\% of the full-correction gain"
                         if ratio is not None and recovered is not None else 'The recovered fractions were undefined because the full-correction gain was nonpositive')
        paragraph = (
            r'With the unchanged checkpoint and fixed $G_S(20\%)$ gallery, we compared no correction, the eight '
            r'ear-hole/tear corrections, eight iterative margin-ratio corrections, and full correction on '
            f"{cfg['n_query_sightings']:,} held-out sightings, preserving the preceding experiment's "
            r'\texttt{ele-SEEK} values including unknowns, correcting each image before retrieval, and averaging hits '
            r'within and then equally across sightings; margin-ratio recomputed its scores after each correction '
            r'and applied the selected attribute to every image in the sighting. '
            f"Recall@1 was {none['recall_at_1_pct']:.2f}\\%, {ear['recall_at_1_pct']:.2f}\\%, "
            f"{margin['recall_at_1_pct']:.2f}\\%, and {full['recall_at_1_pct']:.2f}\\%, respectively; "
            f"margin-ratio improved over no correction by {margin['improvement_pp']:.2f} percentage points "
            f"(paired 95\\% CI [{margin['ci95_low_pp']:.2f}, {margin['ci95_high_pp']:.2f}]) and differed from ear-only by "
            f"{contrast['difference_pp']:+.2f} points (95\\% CI [{contrast['ci95_low_pp']:.2f}, {contrast['ci95_high_pp']:.2f}]), "
            f"using {cfg['bootstrap_draws']:,} shared sighting-bootstrap resamples. "
            + recovery_text + r'; review counts were 0, 8, 8, and 16 attributes per image, not measured annotation times.'
        )
    (out / 'section_4_2.tex').write_text(paragraph + '\n')
    print('\n'.join(lines[:15]))


if __name__ == '__main__':
    main()
