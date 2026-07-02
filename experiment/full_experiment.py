"""
增量实验：只运行尚未完成的3个场景
CIFAR-10 + simple_cnn 已完成 → 跳过
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from run_all import run_one_scenario, SEEDS, OUTPUT_DIR_BASE

config = {
    'batch_size': 64,
    'train_subset': 3000,
    'test_subset': 2000,
    'epochs': 20,
    'lr': 0.05,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'seeds': SEEDS,
}

# ← 每个场景输出到独立子文件夹
SCENARIOS = [
    ('CIFAR-10', 'resnet18', 'CIFAR10_resnet18'),
    ('SVHN',     'simple_cnn', 'SVHN_simple_cnn'),
    ('SVHN',     'resnet18',   'SVHN_resnet18'),
]

import run_all as ra
original_output = ra.OUTPUT_DIR_BASE

for ds, mt, folder in SCENARIOS:
    scene_out = os.path.join(OUTPUT_DIR_BASE, folder)
    os.makedirs(os.path.join(scene_out, 'data'), exist_ok=True)
    os.makedirs(os.path.join(scene_out, 'figures'), exist_ok=True)

    # 切换输出目录到场景子文件夹
    ra.OUTPUT_DIR_BASE = scene_out
    cfg = {**config, 'model_type': mt, 'output_dir': scene_out}

    print(f"\n{'*'*60}")
    print(f"* SCENARIO: {ds} + {mt}")
    print(f"* Output: {scene_out}")
    print(f"{'*'*60}")
    result = run_one_scenario(ds, mt, cfg)

    # 保存场景JSON
    import json
    import numpy as np
    scene_data = {
        'dataset': result['dataset'],
        'model_type': result['model_type'],
        'config': {k: v for k, v in cfg.items() if k != 'output_dir'},
        'all_results': [
            {k: v for k, v in r.items()
             if k not in ['history', 'gradient_stats', 'train_loss', 'test_loss',
                          'train_acc', 'test_acc', 'epoch_times']}
            for r in result['all_results']
        ],
        'statistics': result['statistics'],
    }
    json_path = os.path.join(scene_out, 'data', 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved: {json_path}")

# 恢复
ra.OUTPUT_DIR_BASE = original_output
print("\nALL 3 SCENARIOS COMPLETE")
