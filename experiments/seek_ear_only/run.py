#!/usr/bin/env python3
"""Fixed-checkpoint ear-only comparison, with mandatory prior endpoint audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import sys

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.seek_intervention_top20 import run_prioritization as base
from experiments.seek_intervention_prioritization.policies import copy_attribute_from_expert


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ear_attributes():
    # Resolve names and categorical slices from the actual dataset encoder schema.
    names = [n for n in base.SEEK.attribute_names if re.fullmatch(r'[LR]_(hole|tear)_[12]', n)]
    assert len(names) == 8 and len(base.SEEK.attribute_names) == 16
    assert all(n in base.SEEK.mappings for n in names)
    return names


def corrected(predicted, expert, names):
    result = predicted.clone()
    for name in names:
        result = copy_attribute_from_expert(result, expert, base.SEEK.attribute_names.index(name))
    return result


def sighting_scores(hits, encounter_ids):
    groups = base.group_positions(encounter_ids)
    return np.asarray([hits[pos].mean(axis=0) for pos in groups.values()], dtype=np.float64)


def paired_bootstrap(scores, draws, seed):
    rng = np.random.default_rng(seed)
    means = np.empty((draws, scores.shape[1]))
    for start in range(0, draws, 100):
        sampled = rng.integers(0, len(scores), size=(min(100, draws-start), len(scores)))
        means[start:start+len(sampled)] = scores[sampled].mean(axis=1) * 100
    return means


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'experiments/seek_intervention_top20/results')
    p.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'results')
    p.add_argument('--prepare-only', action='store_true', help='Collect fixed features and audit preceding endpoints only.')
    p.add_argument('--train-indices', type=Path, help='Relocated original split file; no resampling.')
    p.add_argument('--test-indices', type=Path, help='Relocated original split file; checked against saved query indices.')
    p.add_argument('--preprocessed-root', type=Path, help='Relocated original mara_preprocessed directory; preserve file suffixes.')
    p.add_argument('--correction-source', choices=['sighting', 'image', 'both'])
    p.add_argument('--bootstrap-draws', type=int, default=10000)
    p.add_argument('--bootstrap-seed', type=int, default=2026)
    args = p.parse_args()
    if not args.prepare_only and args.correction_source is None:
        p.error('--correction-source is required: the preceding experiment used ele-SEEK, not subject-SEEK')
    source, out = args.source.resolve(), args.output_dir.resolve()
    assert source != out
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((source / 'prioritization_results.json').read_text())
    cfg = saved['config']
    train_path = args.train_indices or ROOT / cfg['train_indices_path']
    test_path = args.test_indices or ROOT / cfg['test_indices_path']
    assert cfg['gallery_correction_fn'] == 'sighting_0.2'
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    dataset = base.EleHandler(subset=cfg['subset'], dataset_type=cfg['dataset'])
    if args.preprocessed_root:
        def relocated(value):
            if pd.isna(value):
                return value
            prefix, marker, suffix = str(value).partition('mara_preprocessed/')
            if not marker:
                raise ValueError(f'Unexpected preprocessed path: {value}')
            return str(args.preprocessed_root.resolve() / suffix)
        for column in ['preprocessed_image_path', 'left_ear_path', 'right_ear_path']:
            dataset.dictonary[column] = dataset.dictonary[column].map(relocated)
    checkpoint = base.resolve_checkpoint_dir(cfg, download=False)
    paths = [checkpoint / n for n in cfg['heatmap_reference']['checkpoint_files']]
    paths += [train_path, test_path]
    paths += [base.METHODS_DIR / n for n in ['data_handler.py', 'seek_code.py', 'retrieval.py', 'backbone.py', 'concept_head_tunneled.py', 'projector.py']]
    paths += [Path(base.__file__), ROOT / 'experiments/seek_intervention_prioritization/policies.py',
              source / 'prioritization_results.json', source / 'query_metadata.npz', source / 'trace_margin_42.npz',
              base.METHODS_DIR.parent / 'data/mara/image_dictionary_optimized.csv']
    paths += [ROOT / 'weights' / (cfg['heatmap_reference']['config'][k] + '.pt')
              for k in ['backbone_for_concepts_pretraining', 'backbone_pretraining', 'concept_head_pretraining']]
    provenance = {str(f): digest(f) for f in paths}
    write_json(out / 'command.json', {'argv': sys.argv, 'cwd': str(ROOT)})
    with wandb.init(mode='disabled'):
        projector, cb, backbone, head = base.build_models(cfg, checkpoint, dataset)
        cache_path = out / 'feature_cache.pt'
        if cache_path.exists():
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
            assert cache['provenance'] == provenance, 'Stale features: input hashes changed'
            train, query = [base.SplitCache(**cache[k]) for k in ['train', 'query']]
        else:
            train_idx, test_idx = base.load_indices(train_path, test_path)
            caches = []
            for label, indices in [('gallery', train_idx), ('query', test_idx)]:
                print(f'Collecting {label}: {len(indices)} images', flush=True)
                loader = DataLoader(Subset(dataset, indices), batch_size=cfg['batch_size'], shuffle=False,
                                    num_workers=cfg['num_workers'], pin_memory=True)
                caches.append(base.collect_cache(loader, dataset, cb, backbone, head, projector.device))
            train, query = caches
            query = query.subset(base.select_query_positions(query, None, cfg['seed']))
            torch.save(dict(provenance=provenance, train=asdict(train), query=asdict(query)), cache_path)
        ref_meta = np.load(source / 'query_metadata.npz')
        for name, value in [('indices', query.indices.numpy()), ('labels', query.labels.numpy()),
                            ('encounter_ids', np.asarray(query.encounter_ids))]:
            np.testing.assert_array_equal(value, ref_meta[name])
        gallery_fn = base.SEEK.get_correction_policy(cfg['gallery_correction_fn'], dataset, default_scope='sighting')
        gallery = base.embeddings_from_cache(projector, train, gallery_fn).to(projector.device)
        retrieval = base.Retrieval(experiment_code='seek_ear_only')
        predicted = base.SEEK.closest_valid_one_hot(query.concept_logits.to(projector.device)).cpu()
        def evaluate(seeks):
            result = base.evaluate_budget(projector, retrieval, query, seeks, gallery, train.labels,
                                          saved['temperature']['temperature'], cfg['epsilon'])
            return result['_hits'], result['recall_at_1']
        none, none_image_mean = evaluate(predicted)
        prior_full, full_image_mean = evaluate(query.sighting_seeks)
        reference = np.load(source / 'trace_margin_42.npz')['hits'][:, [0, 16]]
        current = np.column_stack([none, prior_full])
        checks = dict(image_hit_mismatches=(current != reference).sum(axis=0).tolist(),
                      image_weighted_recall_pct=[none_image_mean, full_image_mean],
                      reference_image_weighted_recall_pct=[torch.tensor(reference[:, i]).float().mean().item()*100 for i in range(2)],
                      sighting_averaged_recall_pct=(sighting_scores(current, query.encounter_ids).mean(0)*100).tolist(),
                      reference_sighting_averaged_recall_pct=(sighting_scores(reference, query.encounter_ids).mean(0)*100).tolist(),
                      tolerance_pp=1e-6, query_order_exact_match=True)
        checks['passed'] = bool(np.array_equal(current, reference) and np.max(np.abs(
            np.array(checks['image_weighted_recall_pct']) - np.array(checks['reference_image_weighted_recall_pct']))) <= 1e-6)
        write_json(out / 'endpoint_audit.json', checks)
        write_json(out / 'provenance.json', provenance)
        print('Endpoint audit:', json.dumps(checks), flush=True)
        assert checks['passed'], 'Endpoint discrepancy: investigate before comparison'
        np.savez_compressed(out / 'query_metadata.npz', indices=query.indices.numpy(), labels=query.labels.numpy(), encounter_ids=query.encounter_ids)
        np.savez_compressed(out / 'gallery_metadata.npz', indices=train.indices.numpy(), labels=train.labels.numpy(), encounter_ids=train.encounter_ids,
                            corrected=np.asarray([s in gallery_fn.corrected_units for s in train.encounter_ids]))
        if args.prepare_only:
            return
        names = ear_attributes()
        sources = ['sighting', 'image'] if args.correction_source == 'both' else [args.correction_source]
        config = dict(reference_config=cfg, correction_sources=sources, ear_attributes=names,
                      attribute_schema=base.SEEK.mappings, bootstrap_draws=args.bootstrap_draws,
                      bootstrap_seed=args.bootstrap_seed, bootstrap_unit='query sighting',
                      aggregation='image retrieval hits -> within-sighting mean -> equal mean over sightings',
                      unknown_handling='copy complete categorical one-hot slice, including underscore; no masking or imputation',
                      n_gallery_images=len(train.labels), n_query_images=len(query.labels),
                      n_query_sightings=len(base.group_positions(query.encounter_ids)),
                      n_query_identities=len(torch.unique(query.labels)), alpha=float(base.alpha_value(projector)),
                      torch_version=torch.__version__, numpy_version=np.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), checkpoint=str(checkpoint),
                      resolved_train_indices=str(train_path.resolve()), resolved_test_indices=str(test_path.resolve()),
                      relocated_preprocessed_root=str(args.preprocessed_root.resolve()) if args.preprocessed_root else None,
                      execution_environment={k: os.environ[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'MPLBACKEND'] if k in os.environ})
        write_json(out / 'configuration.json', config)
        all_rows, all_diffs, recovery = [], [], {}
        for correction_source in sources:
            expert = query.sighting_seeks if correction_source == 'sighting' else query.subject_seeks
            ear_seeks = corrected(predicted, expert, names)
            assert torch.equal(corrected(predicted, expert, base.SEEK.attribute_names), expert)
            ear, _ = evaluate(ear_seeks)
            full = prior_full if correction_source == 'sighting' else evaluate(expert)[0]
            hits = np.column_stack([none, ear, full])
            scores = sighting_scores(hits, query.encounter_ids)
            means = scores.mean(axis=0) * 100
            boot = paired_bootstrap(scores, args.bootstrap_draws, args.bootstrap_seed)
            conditions = ['none', 'ear', 'full']
            np.savez_compressed(out / f'image_hits_{correction_source}.npz', hits=hits, conditions=conditions)
            records = []
            for j, (sighting, pos) in enumerate(base.group_positions(query.encounter_ids).items()):
                records.append(dict(sighting=sighting, identity=query.ele_ids[pos[0]], n_images=len(pos),
                                    **{f'recall_at_1_{c}': scores[j, i] for i, c in enumerate(conditions)},
                                    **{f'hits_{c}': int(hits[pos, i].sum()) for i, c in enumerate(conditions)}))
            pd.DataFrame(records).to_csv(out / f'per_sighting_{correction_source}.csv', index=False)
            for i, (condition, count) in enumerate(zip(conditions, [0, 8, 16])):
                lo, hi = np.percentile(boot[:, i]-boot[:, 0], [2.5, 97.5])
                all_rows.append(dict(correction_source=correction_source, condition=condition, attributes_reviewed=count,
                                     recall_at_1_pct=means[i], improvement_pp=means[i]-means[0], ci95_low_pp=lo, ci95_high_pp=hi))
            for a, b in [(1, 0), (2, 0), (1, 2)]:
                lo, hi = np.percentile(boot[:, a]-boot[:, b], [2.5, 97.5])
                all_diffs.append(dict(correction_source=correction_source, contrast=f'{conditions[a]} - {conditions[b]}',
                                      difference_pp=means[a]-means[b], ci95_low_pp=lo, ci95_high_pp=hi))
            denominator = means[2]-means[0]
            recovery[correction_source] = dict(fraction_of_full_improvement_pct=float(100*(means[1]-means[0])/denominator) if denominator > 0 else None,
                                               undefined_reason=None if denominator > 0 else 'Full correction does not improve over no correction.')
        pd.DataFrame(all_rows).to_csv(out / 'summary.csv', index=False)
        pd.DataFrame(all_diffs).to_csv(out / 'paired_differences.csv', index=False)
        write_json(out / 'summary.json', dict(rows=all_rows, paired_differences=all_diffs, recovery=recovery))
        env_args = [f'{k}={v}' for k, v in config['execution_environment'].items()]
        command = 'cd ' + shlex.quote(str(ROOT)) + '\n' + shlex.join(['env', *env_args, sys.executable, *sys.argv]) + '\n'
        if sources == ['sighting']:
            command += shlex.join(['env', *env_args, sys.executable, 'experiments/seek_ear_only/add_margin.py', '--results-dir', str(out)]) + '\n'
            if out == Path(__file__).parent.resolve() / 'results':
                command += shlex.join(['env', *env_args, sys.executable, 'experiments/seek_ear_only/image_weighted.py']) + '\n'
                command += shlex.join(['env', *env_args, sys.executable, 'experiments/seek_ear_only/comparison_texts.py']) + '\n'
            command += shlex.join([sys.executable, 'experiments/seek_ear_only/report.py', '--results-dir', str(out)]) + '\n'
        (out / 'reproduce.sh').write_text('#!/bin/bash\nset -euo pipefail\n' + command)
        print(pd.DataFrame(all_rows).to_string(index=False), flush=True)
        print(json.dumps(recovery), flush=True)


if __name__ == '__main__':
    main()
