#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path('/hpc2hdd/home/rmeng603/workspace/MFANNS')
EXP_DIR = ROOT / 'simulator/memory/20260329/mfnns_t2i_k10_ef20_30_40_dualq_q20_100'
YAML_DIR = EXP_DIR / 'yamls'
LOG_DIR = EXP_DIR / 'logs'
MANIFEST_PATH = EXP_DIR / 'case_manifest.tsv'
BASE_YAML = ROOT / 'simulator/memory/20260326/multidataset_nine_config_fp16_hotrep_cfs/yamls/t2i1m_normalized_mfnns_etopt_hotrep.yaml'

DATASET = 't2i1m'
VARIANT = 'normalized'
K_NEIGHBORS = 10
EF_CANDIDATES = [20, 30, 40]
QUEUE_CANDIDATES = list(range(20, 101))

MANIFEST_FIELDS = [
    'case_order',
    'dataset',
    'variant',
    'case_name',
    'ef_search',
    'queue_size',
    'warmup_size',
    'k_neighbors',
    'yaml_path',
    'stats_path',
    'base_yaml',
    'note',
]


def load_yaml(path: Path) -> dict:
    with path.open() as fin:
        return yaml.safe_load(fin)


def dump_yaml(data: dict, path: Path) -> None:
    with path.open('w') as fout:
        yaml.safe_dump(data, fout, sort_keys=False)


def configure_yaml(data: dict, ef_search: int, queue_size: int, stats_path: Path) -> None:
    front = data['Frontend']
    front['ef_search'] = ef_search
    front['k_neighbors'] = K_NEIGHBORS
    front['dualQueueLowerBoundQueueSize'] = queue_size
    front['dualQueueLowerBoundWarmupSize'] = queue_size - 1
    front['stat_path'] = str(stats_path)
    front['perQuerySummaryEnable'] = False
    front['debugIssueTraceEnable'] = False
    front['debugDuplicateAcceptEnable'] = False


def main() -> None:
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    base = load_yaml(BASE_YAML)
    rows: list[dict[str, str]] = []
    case_order = 0
    for ef_search in EF_CANDIDATES:
        for queue_size in QUEUE_CANDIDATES:
            case_order += 1
            case_name = f'{DATASET}_{VARIANT}_k{K_NEIGHBORS}_ef{ef_search}_q{queue_size}'
            yaml_path = YAML_DIR / f'{case_name}.yaml'
            stats_path = LOG_DIR / f'{case_name}_stats.yml'
            data = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
            configure_yaml(data, ef_search, queue_size, stats_path)
            dump_yaml(data, yaml_path)
            rows.append({
                'case_order': str(case_order),
                'dataset': DATASET,
                'variant': VARIANT,
                'case_name': case_name,
                'ef_search': str(ef_search),
                'queue_size': str(queue_size),
                'warmup_size': str(queue_size - 1),
                'k_neighbors': str(K_NEIGHBORS),
                'yaml_path': str(yaml_path),
                'stats_path': str(stats_path),
                'base_yaml': str(BASE_YAML),
                'note': 'based_on=20260329/mfnns_etopt_hotrep_k10_ef_sweep;t2i;warmup=q-1',
            })

    with MANIFEST_PATH.open('w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=MANIFEST_FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f'generated_cases\t{len(rows)}')
    print(f'manifest\t{MANIFEST_PATH}')


if __name__ == '__main__':
    main()
