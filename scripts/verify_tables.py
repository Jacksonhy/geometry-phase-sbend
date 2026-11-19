"""Recompute saved summary arithmetic; does not run FDTD, training or inference."""
import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    table = args.tables
    errors = []
    groups = collections.defaultdict(list)
    for r in rows(table / 'surrogate_endpoints_per_geometry.csv'):
        groups[r['endpoint'], r['seed'], r['method']].append(r)
    seed_groups = collections.defaultdict(list)
    for r in rows(table / 'surrogate_main_table_per_seed.csv'):
        rr = groups[r['endpoint'], r['seed'], r['method']]
        assert len(rr) == len({v['geometry_id'] for v in rr}) == 72
        mean = statistics.mean(float(v['value']) for v in rr)
        errors.append(abs(mean - float(r['mean'])))
        seed_groups[r['endpoint'], r['method']].append(mean)
    euler = {}
    for r in rows(table / 'surrogate_main_table_three_seed_summary.csv'):
        vv = seed_groups[r['endpoint'], r['method']]
        assert len(vv) == 3
        errors.extend([abs(statistics.mean(vv) - float(r['mean_of_seed_means'])),
                       abs(statistics.stdev(vv) - float(r['sample_sd_of_seed_means']))])
        if r['endpoint'] == 'whole_rel_l2':
            euler[r['method']] = statistics.mean(vv)
    sine = rows(table / 'sine_blind_per_geometry.csv')
    mappings = [
        ('whole_RelL2', 'field', 'whole', 0.0, 'complex_rel_l2'),
        ('bend_RelL2', 'field', 'bend_core', 0.0, 'complex_rel_l2'),
        ('bend_phase_deg', 'field', 'bend_core', 0.0, 'phase_wmae_deg'),
        ('normal_phase_deg', 'pair', 'normal_pair_all', .5, 'normal_relative_phase_wmae_deg'),
        ('cross_RelL2', 'pair', 'normal_pair_all', .5, 'coherence_complex_rel_l2'),
    ]
    sine_means = collections.defaultdict(list)
    sine_seed_comparisons = 0
    for r in rows(table / 'sine_seed_summary.csv'):
        for name, kind, region, eta, metric in mappings:
            rr = [v for v in sine if v['seed'] == r['seed'] and v['method'] == r['method']
                  and v['kind'] == kind and v['region'] == region and float(v['eta']) == eta
                  and float(v['gate']) == .05 and v['convention'] == 'raw'
                  and int(v['truth_resolution']) == 20]
            assert len(rr) == len({v['geometry_id'] for v in rr}) == 36, (name, r, len(rr))
            mean = statistics.mean(float(v[metric]) for v in rr)
            errors.append(abs(mean - float(r[name])))
            sine_means[r['method'], name].append(mean)
            sine_seed_comparisons += 1
    for r in rows(table / 'sine_main_table.csv'):
        for name, *_ in mappings:
            errors.append(abs(statistics.mean(sine_means[r['method'], name]) - float(r[name])))
    timings = []
    for r in rows(table / 'surrogate_online_cost_summary.csv'):
        errors.append(abs(float(r['fdtd_reference_seconds']) / float(r['query_seconds']) - float(r['online_speed_ratio'])))
        errors.append(abs(float(r['reconstructed_offline_seconds']) - float(r['offline_measured_subtotal_seconds']) - float(r['reconstructed_overhead_seconds'])))
        if r['method'] == 'local_tangent':
            n = math.ceil((float(r['reconstructed_offline_seconds']) + float(r['model_session_setup_seconds'])) /
                          (float(r['fdtd_reference_seconds']) - float(r['query_seconds'])))
            timings.append(dict(mode=r['mode'], seconds=float(r['query_seconds']),
                                break_even_queries=n, full_72_batches_queries=math.ceil(n / 72) * 72))
    assert max(errors) < 1e-10, max(errors)
    result = dict(passed=True, scope='Arithmetic on archived geometry metrics only; not independent field verification',
                  numerical_comparisons=len(errors), max_absolute_difference=max(errors),
                  euler_72_whole_field_means=euler, sine_seed_endpoint_comparisons=sine_seed_comparisons,
                  local_tangent_timing=timings, FDTD_runs=0, training_runs=0, inference_runs=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
