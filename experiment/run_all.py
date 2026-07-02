"""
主实验运行脚本 V2
批量归一化层位置对梯度稳定性的影响：基于多数据集+多架构的交叉验证

支持:
- 数据集: CIFAR-10 (本地) / SVHN (torchvision下载)
- 架构: SimpleCNN / ResNet18Light
- BN配置: NoBN / BN_pre (Conv→BN→ReLU) / BN_post (Conv→ReLU→BN)
- 统计检验: 配对t检验 + Cohen's d + ANOVA
"""

import sys
import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))

from models import ResNet18Light, SimpleCNN, create_model_configs, count_parameters
from trainer import train_model, compute_gradient_stability_metrics
from mediation import run_mediation_analysis
from visualize import generate_all_figures

# ============================================================
# 全局配置
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]
MODEL_CONFIGS_TO_TEST = ['NoBN', 'BN_pre', 'BN_post']

# CIFAR-10 标准化参数
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
# SVHN 标准化参数 (预计算值)
SVHN_MEAN = np.array([0.4377, 0.4438, 0.4728], dtype=np.float32).reshape(1, 3, 1, 1)
SVHN_STD  = np.array([0.1980, 0.2010, 0.1970], dtype=np.float32).reshape(1, 3, 1, 1)

OUTPUT_DIR_BASE = os.path.join(os.path.dirname(__file__), 'output')
CIFAR10_PATH = os.path.join(os.path.dirname(__file__), '..', 'cifar-10-batches-py')


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 数据加载
# ============================================================

def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从本地文件加载CIFAR-10"""
    def unpickle(file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        X_train_list.append(batch[b'data'])
        y_train_list.extend(batch[b'labels'])

    X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_train = X_train.reshape(-1, 3, 32, 32) / 255.0

    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = test_batch[b'data'].astype(np.float32)
    y_test = np.array(test_batch[b'labels'], dtype=np.int64)
    X_test = X_test.reshape(-1, 3, 32, 32) / 255.0

    return X_train, y_train, X_test, y_test


def get_dataloaders(dataset_name: str, batch_size: int = 256,
                    train_size: int = None, test_size: int = None,
                    augment: bool = True):
    """
    统一数据加载接口

    Args:
        augment: 是否启用数据增强 (GPU模式开启)

    Returns:
        train_loader, test_loader, num_classes
    """
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_root, exist_ok=True)
    rng = np.random.RandomState(42)

    if dataset_name == 'CIFAR-10':
        X_train, y_train, X_test, y_test = load_cifar10(CIFAR10_PATH)
        num_classes = 10

        # 先子集采样再增强 (避免对全量50K做增强)
        if train_size is not None and train_size < len(y_train):
            indices = rng.choice(len(y_train), train_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        # 数据增强
        if augment:
            n_aug = X_train.shape[0]
            pad = 4
            # Padding到40×40
            padded = np.pad(X_train, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='reflect')
            # 向量化 RandomCrop: 随机左上角
            crop_x = rng.randint(0, 9, n_aug)
            crop_y = rng.randint(0, 9, n_aug)
            X_aug = np.empty_like(X_train)
            for i in range(n_aug):
                X_aug[i] = padded[i, :, crop_y[i]:crop_y[i]+32, crop_x[i]:crop_x[i]+32]
            # RandomHorizontalFlip (50%)
            flip = rng.rand(n_aug) < 0.5
            X_aug[flip] = X_aug[flip, :, :, ::-1]
            X_train = X_aug

        X_train = (X_train - CIFAR10_MEAN) / CIFAR10_STD
        X_test  = (X_test - CIFAR10_MEAN) / CIFAR10_STD

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_test_t  = torch.from_numpy(X_test)
        y_test_t  = torch.from_numpy(y_test)

        full_train = TensorDataset(X_train_t, y_train_t)
        full_test  = TensorDataset(X_test_t, y_test_t)

    elif dataset_name == 'SVHN':
        train_transforms = []
        if augment:
            train_transforms.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])
        train_transform = transforms.Compose(train_transforms)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])
        full_train = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        full_test  = datasets.SVHN(data_root, split='test', download=True, transform=test_transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 子集采样 (CIFAR-10已在上面完成采样，此处仅针对SVHN等)
    if dataset_name != 'CIFAR-10':
        if train_size is not None:
            indices = rng.choice(len(full_train), min(train_size, len(full_train)), replace=False)
            train_dataset = Subset(full_train, indices)
        else:
            train_dataset = full_train
    else:
        train_dataset = full_train  # CIFAR-10已子集化

    if test_size is not None:
        indices = rng.choice(len(full_test), min(test_size, len(full_test)), replace=False)
        test_dataset = Subset(full_test, indices)
    else:
        test_dataset = full_test

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"  Dataset: {dataset_name} | Augment: {augment}")
    print(f"    Train: {len(train_dataset)}, Test: {len(test_dataset)}, Batch: {batch_size}")

    return train_loader, test_loader, num_classes


# ============================================================
# 单次实验
# ============================================================

def run_single(model_config_name: str, seed: int, train_loader: DataLoader,
               test_loader: DataLoader, config: Dict, device: torch.device) -> Dict:
    set_seed(seed)

    model_configs = create_model_configs(
        model_type=config['model_type'],
        num_classes=config['num_classes']
    )
    model = model_configs[model_config_name].to(device)
    n_params = count_parameters(model)

    train_cfg = {
        'epochs': config['epochs'],
        'lr': config['lr'],
        'seed': seed,
        'model_config': model_config_name,
    }

    history = train_model(model, train_loader, test_loader, train_cfg, device, verbose=False)
    sm = compute_gradient_stability_metrics(history['gradient_stats'])

    return {
        'use_bn': 'BN' in model_config_name,
        'model_config': model_config_name,
        'seed': seed,
        'n_params': n_params,
        'best_acc': history['best_acc'],
        'convergence_epoch': history['convergence_epoch'],
        'convergence_speed': history['best_acc'],
        'gradient_stability': sm['stability_score'],
        'gradient_layer_variance': sm['layer_variance'],
        'gradient_cross_layer_ratio': sm['cross_layer_ratio'],
        'gradient_temporal_variance': sm['temporal_variance'],
        'train_loss': history['train_loss'],
        'test_loss': history['test_loss'],
        'train_acc': history['train_acc'],
        'test_acc': history['test_acc'],
        'gradient_stats': history['gradient_stats'],
        'layer_names': history['layer_names'],
        'epoch_times': history['epoch_times'],
        'history': history,
    }


# ============================================================
# 统计检验
# ============================================================

def run_statistical_tests(all_results: List[Dict], dataset_name: str,
                          model_type: str) -> Dict:
    """
    执行完整统计检验

    1. 配对t检验: BN_pre vs NoBN, BN_post vs NoBN, BN_pre vs BN_post
    2. 单因素ANOVA: 三种配置间比较
    3. Cohen's d: 两两效应量
    """
    results_by_config = defaultdict(list)
    for r in all_results:
        results_by_config[r['model_config']].append(r)

    metrics = ['best_acc', 'gradient_stability', 'gradient_layer_variance',
               'gradient_cross_layer_ratio', 'convergence_epoch']

    tests = {}
    configs = ['NoBN', 'BN_pre', 'BN_post']
    pairs = [('BN_pre', 'NoBN'), ('BN_post', 'NoBN'), ('BN_pre', 'BN_post')]

    # 1. 配对t检验 (过滤NaN)
    for metric in metrics:
        tests[f'{metric}_pairwise'] = {}
        for c1, c2 in pairs:
            v1 = np.array([r[metric] for r in results_by_config[c1]])
            v2 = np.array([r[metric] for r in results_by_config[c2]])
            # 过滤NaN/Inf
            v1 = v1[~np.isnan(v1) & ~np.isinf(v1)]
            v2 = v2[~np.isnan(v2) & ~np.isinf(v2)]
            if len(v1) > 1 and len(v2) > 1:
                t_stat, p_val = scipy_stats.ttest_ind(v1, v2, equal_var=False)
                cohens_d = (np.mean(v1) - np.mean(v2)) / max(1e-15, np.sqrt((np.var(v1) + np.var(v2)) / 2))
                tests[f'{metric}_pairwise'][f'{c1}_vs_{c2}'] = {
                    't_stat': float(t_stat), 'p_value': float(p_val),
                    'cohens_d': float(cohens_d),
                    'mean1': float(np.mean(v1)), 'std1': float(np.std(v1)),
                    'mean2': float(np.mean(v2)), 'std2': float(np.std(v2)),
                }

    # 2. ANOVA (过滤NaN)
    for metric in metrics:
        groups = []
        for c in configs:
            g = np.array([r[metric] for r in results_by_config[c]])
            g = g[~np.isnan(g) & ~np.isinf(g)]
            if len(g) > 1:
                groups.append(g)
        if len(groups) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            tests[f'{metric}_anova'] = {'f_stat': float(f_stat), 'p_value': float(p_val)}

    return tests


def print_statistical_results(tests: Dict, dataset_name: str, model_type: str):
    """打印统计检验结果"""
    print(f"\n{'='*60}")
    print(f"STATISTICAL TESTS: {dataset_name} + {model_type}")
    print('='*60)

    for metric in ['best_acc', 'gradient_stability', 'gradient_layer_variance']:
        anova_key = f'{metric}_anova'
        if anova_key in tests and tests[anova_key] is not None:
            a = tests[anova_key]
            sig = '***' if a['p_value'] < 0.001 else '**' if a['p_value'] < 0.01 else '*' if a['p_value'] < 0.05 else 'ns'
            print(f"\n  {metric} ANOVA: F = {a['f_stat']:.3f}, p = {a['p_value']:.4f} {sig}")

        pair_key = f'{metric}_pairwise'
        if pair_key in tests and tests[pair_key]:
            for pair_name, pair_data in tests[pair_key].items():
                sig = '***' if pair_data['p_value'] < 0.001 else '**' if pair_data['p_value'] < 0.01 else '*' if pair_data['p_value'] < 0.05 else 'ns'
                mag = 'large' if abs(pair_data['cohens_d']) > 0.8 else 'medium' if abs(pair_data['cohens_d']) > 0.5 else 'small'
                print(f"    {pair_name}: t = {pair_data['t_stat']:.3f}, p = {pair_data['p_value']:.4f} {sig}, d = {pair_data['cohens_d']:.3f} ({mag})")


# ============================================================
# 单场景实验
# ============================================================

def run_one_scenario(dataset_name: str, model_type: str, config: Dict) -> Dict:
    """
    运行一个完整场景: 特定数据集 + 特定架构
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"\n{'#'*60}")
    print(f"# SCENARIO: {dataset_name} + {model_type}")
    print(f"# Device: {device} {'(AMP)' if use_amp else ''} | Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#'*60}")

    # 数据 (GPU模式启用数据增强)
    augment = config.get('augment', use_amp)
    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name, config['batch_size'],
        config['train_subset'], config['test_subset'],
        augment=augment
    )
    config['num_classes'] = num_classes
    config['model_type'] = model_type

    # 运行
    all_results = []
    all_histories = {}
    total = len(MODEL_CONFIGS_TO_TEST) * len(config['seeds'])
    run_idx = 0

    for model_config_name in MODEL_CONFIGS_TO_TEST:
        all_histories[model_config_name] = []
        for seed in config['seeds']:
            run_idx += 1
            result = run_single(model_config_name, seed, train_loader, test_loader, config, device)
            all_results.append(result)
            all_histories[model_config_name].append(result['history'])
            print(f"  [{run_idx}/{total}] {model_config_name:8s} seed={seed:4d} | "
                  f"acc={result['best_acc']:.4f} | stab={result['gradient_stability']:.2f} | "
                  f"cv={result['gradient_layer_variance']:.4f}")

    # 汇总
    print(f"\n  {'─'*50}")
    for mc in MODEL_CONFIGS_TO_TEST:
        cr = [r for r in all_results if r['model_config'] == mc]
        accs = [r['best_acc'] for r in cr]
        stabs = [r['gradient_stability'] for r in cr]
        layers = [r['gradient_layer_variance'] for r in cr]
        ratios = [r['gradient_cross_layer_ratio'] for r in cr]
        print(f"  {mc:8s}: Acc={np.mean(accs):.4f}\u00b1{np.std(accs):.4f}  "
              f"Stab={np.mean(stabs):.0f}\u00b1{np.std(stabs):.0f}  "
              f"CV={np.mean(layers):.4f}  Ratio={np.mean(ratios):.2f}")

    # 统计检验
    stats = run_statistical_tests(all_results, dataset_name, model_type)
    print_statistical_results(stats, dataset_name, model_type)

    # 可视化
    scene_name = f'{dataset_name.replace("-","")}_{model_type}'
    mediation_input = [{'use_bn': r['use_bn'], 'gradient_stability': r['gradient_stability'],
                         'convergence_speed': r['convergence_speed'], 'best_acc': r['best_acc']}
                       for r in all_results]
    mediation_result = run_mediation_analysis(mediation_input)
    mediation_result['all_results'] = mediation_input

    print(f"\n  Generating figures...")
    generate_all_figures(all_histories, mediation_result, scene_name, config.get('output_dir'))
    print(f"  Figures saved.")

    return {
        'dataset': dataset_name,
        'model_type': model_type,
        'all_results': all_results,
        'all_histories': all_histories,
        'statistics': stats,
        'mediation': mediation_result,
    }


# ============================================================
# 主入口：多场景编排
# ============================================================

def run_all_scenarios(config: Dict) -> Dict:
    """
    运行所有场景组合并生成综合报告
    """
    datasets = config.get('datasets', ['CIFAR-10'])
    model_types = config.get('model_types', ['simple_cnn'])

    all_scenarios = []
    total_scenarios = len(datasets) * len(model_types)

    print(f"\n{'#'*70}")
    print(f"# FULL EXPERIMENT: {total_scenarios} scenarios")
    print(f"# Datasets: {datasets}")
    print(f"# Models: {model_types}")
    print(f"# BN Configs: {MODEL_CONFIGS_TO_TEST}")
    print(f"# Seeds: {config['seeds']}")
    print(f"# Total runs: {total_scenarios * len(MODEL_CONFIGS_TO_TEST) * len(config['seeds'])}")
    print(f"# Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    for ds in datasets:
        for mt in model_types:
            scenario = run_one_scenario(ds, mt, config)
            all_scenarios.append(scenario)

    # 生成汇总报告
    generate_summary_report(all_scenarios, config)

    return {'scenarios': all_scenarios}


def generate_summary_report(scenarios: List[Dict], config: Dict):
    """生成跨场景汇总报告"""
    print(f"\n{'='*70}")
    print("CROSS-SCENARIO SUMMARY REPORT")
    print('='*70)

    # 汇总所有场景的主要指标
    for scenario in scenarios:
        ds = scenario['dataset']
        mt = scenario['model_type']
        print(f"\n{'─'*60}")
        print(f"  {ds} + {mt}")
        print(f"{'─'*60}")
        print(f"  {'Config':<10} {'Acc':>10} {'GradStab':>12} {'LayerCV':>10} {'Ratio':>8} {'ConvEp':>8}")
        print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*10} {'─'*8} {'─'*8}")

        all_results = scenario['all_results']
        for mc in MODEL_CONFIGS_TO_TEST:
            cr = [r for r in all_results if r['model_config'] == mc]
            if cr:
                acc = f"{np.mean([r['best_acc'] for r in cr]):.4f}\u00b1{np.std([r['best_acc'] for r in cr]):.4f}"
                stab = f"{np.mean([r['gradient_stability'] for r in cr]):.0f}\u00b1{np.std([r['gradient_stability'] for r in cr]):.0f}"
                lcv = f"{np.mean([r['gradient_layer_variance'] for r in cr]):.4f}"
                rat = f"{np.mean([r['gradient_cross_layer_ratio'] for r in cr]):.2f}"
                cep = f"{np.mean([r['convergence_epoch'] for r in cr]):.1f}"
                print(f"  {mc:<10} {acc:>10} {stab:>12} {lcv:>10} {rat:>8} {cep:>8}")

    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_data_dir = os.path.join(OUTPUT_DIR_BASE, 'data')
    os.makedirs(output_data_dir, exist_ok=True)

    full_results = {
        'config': {k: v for k, v in config.items() if k != 'output_dir'},
        'timestamp': timestamp,
        'scenarios': [
            {
                'dataset': s['dataset'],
                'model_type': s['model_type'],
                'all_results': [
                    {k: v for k, v in r.items()
                     if k not in ['history', 'gradient_stats', 'train_loss', 'test_loss',
                                  'train_acc', 'test_acc', 'epoch_times']}
                    for r in s['all_results']
                ],
                'statistics': s['statistics'],
            }
            for s in scenarios
        ],
        'total_runs': sum(len(s['all_results']) for s in scenarios),
    }

    results_path = os.path.join(output_data_dir, f'full_experiment_results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    # 同时更新最新版本
    latest_path = os.path.join(output_data_dir, 'experiment_results_latest.json')
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nFull results saved to: {results_path}")
    print(f"Latest results saved to: {latest_path}")
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print('='*70)


# ============================================================
# 快速入口
# ============================================================
if __name__ == '__main__':
    config = {
        'datasets': ['CIFAR-10', 'SVHN'],
        'model_types': ['simple_cnn', 'resnet18'],
        'batch_size': 64,
        'train_subset': 3000,
        'test_subset': 2000,
        'epochs': 20,
        'lr': 0.05,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'seeds': SEEDS,
        'output_dir': OUTPUT_DIR_BASE,
    }
    run_all_scenarios(config)
