"""
华为云GPU一键运行脚本
T4/V100 GPU + 混合精度 + 数据增强 + 大规模数据

配置升级:
- GPU: 自动检测CUDA + AMP混合精度
- 数据: 全量训练 + RandomCrop + HorizontalFlip
- Epoch: 40 (从20升级)
- Batch: 256 (从64升级)
- 训练量: ~10x原CPU实验

用法: python run_cloud.py
预计耗时: T4 GPU约25-35分钟完成全部4场景
输出: output/ 目录 (每个场景独立子文件夹)
"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))

from run_all import run_one_scenario, SEEDS

# ============================================================
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = DEVICE.type == 'cuda'

# GPU模式 vs CPU模式配置差异
if USE_GPU:
    CONFIG = {
        'batch_size': 256,
        'train_subset': 10000,    # 10K训练样本 (GPU轻松)
        'test_subset': 5000,       # 5K测试
        'epochs': 40,              # 40 epoch充分收敛
        'lr': 0.05,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'seeds': SEEDS,
        'augment': True,           # 数据增强ON
    }
    gpu_name = torch.cuda.get_device_name(0) if USE_GPU else 'N/A'
    print(f"GPU DETECTED: {gpu_name}")
else:
    CONFIG = {
        'batch_size': 64,
        'train_subset': 3000,
        'test_subset': 2000,
        'epochs': 20,
        'lr': 0.05,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'seeds': SEEDS,
        'augment': False,
    }
    print("WARNING: No GPU detected, falling back to CPU (slower)")

SCENARIOS = [
    ('CIFAR-10', 'simple_cnn', 'CIFAR10_simple_cnn'),
    ('CIFAR-10', 'resnet18',   'CIFAR10_resnet18'),
    ('SVHN',     'simple_cnn', 'SVHN_simple_cnn'),
    ('SVHN',     'resnet18',   'SVHN_resnet18'),
]
# ============================================================

print("=" * 70)
print("BATCH NORMALIZATION POSITION vs GRADIENT STABILITY")
print(f"Device: {DEVICE} | GPU: {USE_GPU}")
print(f"Scenarios: {len(SCENARIOS)} | Configs per: 3 | Seeds: {len(SEEDS)}")
print(f"Total training runs: {len(SCENARIOS)*3*len(SEEDS)}")
print(f"Total samples/run: {CONFIG['train_subset']} train + {CONFIG['test_subset']} test")
print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | Augment: {CONFIG['augment']}")
print("=" * 70)

all_scene_results = []
import time as _time
_start = _time.time()

for idx, (ds, mt, folder) in enumerate(SCENARIOS):
    scene_out = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(os.path.join(scene_out, 'data'), exist_ok=True)
    os.makedirs(os.path.join(scene_out, 'figures'), exist_ok=True)

    cfg = {**CONFIG, 'model_type': mt, 'output_dir': scene_out}

    print(f"\n{'#'*60}")
    print(f"# [{idx+1}/{len(SCENARIOS)}] {ds} + {mt}")
    print(f"# Output: {scene_out}")
    print(f"{'#'*60}")

    scene_start = _time.time()
    result = run_one_scenario(ds, mt, cfg)
    scene_elapsed = _time.time() - scene_start

    scene_data = {
        'dataset': ds, 'model_type': mt,
        'config': {k: v for k, v in cfg.items() if k != 'output_dir'},
        'all_results': [
            {k: v for k, v in r.items()
             if k not in ['history', 'gradient_stats', 'train_loss',
                          'test_loss', 'train_acc', 'test_acc', 'epoch_times']}
            for r in result['all_results']
        ],
        'statistics': result['statistics'],
    }
    with open(os.path.join(scene_out, 'data', 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2, ensure_ascii=False, default=str)
    all_scene_results.append(scene_data)
    print(f"  >> DONE in {scene_elapsed:.1f}s. Saved to {scene_out}")

# ============================================================
# 全局汇总
# ============================================================
total_elapsed = _time.time() - _start
print(f"\n{'='*70}")
print(f"GENERATING GLOBAL SUMMARY (total time: {total_elapsed:.1f}s)")
print("=" * 70)

summary = {
    'device': str(DEVICE),
    'gpu': USE_GPU,
    'gpu_name': gpu_name if USE_GPU else 'CPU',
    'total_runs': sum(len(s['all_results']) for s in all_scene_results),
    'scenarios': len(all_scene_results),
    'total_time_seconds': total_elapsed,
    'results_by_scenario': {},
}

for s in all_scene_results:
    key = f"{s['dataset']}+{s['model_type']}"
    by_cfg = {}
    for cfg_name in ['NoBN', 'BN_pre', 'BN_post']:
        cr = [r for r in s['all_results'] if r['model_config'] == cfg_name]
        if cr:
            by_cfg[cfg_name] = {
                'accuracy': np.mean([r['best_acc'] for r in cr]),
                'accuracy_std': np.std([r['best_acc'] for r in cr]),
                'gradient_stability': np.mean([r['gradient_stability'] for r in cr]),
                'gradient_stability_std': np.std([r['gradient_stability'] for r in cr]),
                'layer_cv': np.mean([r['gradient_layer_variance'] for r in cr]),
                'cross_layer_ratio': np.mean([r['gradient_cross_layer_ratio'] for r in cr]),
                'convergence_epoch': np.mean([r['convergence_epoch'] for r in cr]),
            }
    summary['results_by_scenario'][key] = by_cfg

with open(os.path.join(OUTPUT_ROOT, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

# 打印汇总
print(f"\n{'='*70}")
print("FINAL SUMMARY TABLE")
print("=" * 70)
for scene_key, cfg_data in summary['results_by_scenario'].items():
    print(f"\n--- {scene_key} ---")
    header = f"  {'Config':<10} {'Accuracy':>13} {'GradStab':>13} {'LayerCV':>9} {'Ratio':>7} {'ConvEp':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cfg_name, m in cfg_data.items():
        acc = f"{m['accuracy']:.4f}\u00b1{m['accuracy_std']:.4f}"
        stab = f"{m['gradient_stability']:.0f}\u00b1{m['gradient_stability_std']:.0f}"
        print(f"  {cfg_name:<10} {acc:>13} {stab:>13} {m['layer_cv']:>9.4f} {m['cross_layer_ratio']:>7.2f} {m['convergence_epoch']:>7.1f}")

print(f"\n{'='*70}")
print(f"ALL DONE! Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
print(f"Output: {OUTPUT_ROOT}")
print("Zip the 'output/' folder and download it.")
print("=" * 70)
