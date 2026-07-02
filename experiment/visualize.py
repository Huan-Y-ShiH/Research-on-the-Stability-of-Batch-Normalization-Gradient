"""
可视化模块
生成SCI论文级图表，中英文兼容
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple, Optional
import os
import json

# 全局样式设置 - SCI期刊标准
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 颜色方案 (ColorBrewer qualitative + 学术配色)
COLORS = {
    'nobn': '#E74C3C',      # 红色 - 无BN
    'bn_pre': '#2980B9',    # 蓝色 - BN前置
    'bn_post': '#27AE60',   # 绿色 - BN后置
    'primary': '#2C3E50',   # 深蓝灰
    'secondary': '#7F8C8D', # 灰色
    'highlight': '#E67E22', # 橙色高亮
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiment', 'output', 'figures')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _label_to_english(config_name: str) -> str:
    """将配置名转为英文标签"""
    mapping = {
        'NoBN': 'w/o BN',
        'BN_pre': 'BN (pre-activation)',
        'BN_post': 'BN (post-activation)',
    }
    return mapping.get(config_name, config_name)


def _get_color(config_name: str) -> str:
    mapping = {'NoBN': COLORS['nobn'], 'BN_pre': COLORS['bn_pre'], 'BN_post': COLORS['bn_post']}
    return mapping.get(config_name, COLORS['secondary'])


def plot_loss_curves(histories: Dict[str, List[Dict]], dataset_name: str = 'MNIST'):
    """
    图1: 训练/测试损失曲线对比

    展示有/无BN对收敛速度的影响
    """
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for config_name, seed_histories in histories.items():
        # 汇总多种子结果
        all_train_loss = np.array([h['train_loss'] for h in seed_histories])
        all_test_loss = np.array([h['test_loss'] for h in seed_histories])

        train_mean = all_train_loss.mean(axis=0)
        train_std = all_train_loss.std(axis=0)
        test_mean = all_test_loss.mean(axis=0)
        test_std = all_test_loss.std(axis=0)

        epochs = np.arange(1, len(train_mean) + 1)
        color = _get_color(config_name)
        label = _label_to_english(config_name)

        # 训练损失
        axes[0].plot(epochs, train_mean, color=color, linewidth=1.5, label=label)
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std,
                             alpha=0.15, color=color)

        # 测试损失
        axes[1].plot(epochs, test_mean, color=color, linewidth=1.5, label=label)
        axes[1].fill_between(epochs, test_mean - test_std, test_mean + test_std,
                             alpha=0.15, color=color)

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend(frameon=True, fancybox=True, shadow=False)

    axes[0].set_ylabel('Training Loss')
    axes[0].set_title(f'(a) Training Loss Convergence ({dataset_name})')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title(f'(b) Test Loss Convergence ({dataset_name})')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig1_loss_curves_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig1_loss_curves_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig1: Loss curves saved for {dataset_name}")


def plot_accuracy_curves(histories: Dict[str, List[Dict]], dataset_name: str = 'MNIST'):
    """
    图2: 准确率收敛曲线
    """
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for config_name, seed_histories in histories.items():
        all_train_acc = np.array([h['train_acc'] for h in seed_histories])
        all_test_acc = np.array([h['test_acc'] for h in seed_histories])

        train_mean = all_train_acc.mean(axis=0)
        train_std = all_train_acc.std(axis=0)
        test_mean = all_test_acc.mean(axis=0)
        test_std = all_test_acc.std(axis=0)

        epochs = np.arange(1, len(train_mean) + 1)
        color = _get_color(config_name)
        label = _label_to_english(config_name)

        axes[0].plot(epochs, train_mean, color=color, linewidth=1.5, label=label)
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std,
                             alpha=0.15, color=color)

        axes[1].plot(epochs, test_mean, color=color, linewidth=1.5, label=label)
        axes[1].fill_between(epochs, test_mean - test_std, test_mean + test_std,
                             alpha=0.15, color=color)

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend(frameon=True, fancybox=True, loc='lower right')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    axes[0].set_title(f'(a) Training Accuracy ({dataset_name})')
    axes[1].set_title(f'(b) Test Accuracy ({dataset_name})')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig2_accuracy_curves_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig2_accuracy_curves_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig2: Accuracy curves saved for {dataset_name}")


def plot_gradient_stability(histories: Dict[str, List[Dict]], dataset_name: str = 'MNIST'):
    """
    图3: 梯度稳定性对比

    核心图 - 展示BN如何影响各层梯度分布
    (a) 各层梯度方差对比 (箱线图)
    (b) 层间梯度范数比对比
    (c) 梯度方差随epoch变化曲线
    """
    ensure_output_dir()
    fig = plt.figure(figsize=(16, 10))

    # 子图(a): 各层梯度方差 - 取第一个种子的最后一次epoch作为代表
    ax1 = fig.add_subplot(2, 3, (1, 2))
    configs = list(histories.keys())
    x_positions = []
    labels = []
    all_variances = []
    colors_list = []

    for ci, config_name in enumerate(configs):
        seed_histories = histories[config_name]
        # 取最后一个种子、最后一个epoch的各层方差
        layer_names = seed_histories[0].get('layer_names', [])
        grad_stats = seed_histories[0].get('gradient_stats', {})

        variances = []
        for name in layer_names:
            if name in grad_stats:
                variances.append(grad_stats[name]['mean_var'])
            else:
                variances.append(0)

        all_variances.append(variances)
        labels.append(_label_to_english(config_name))
        colors_list.append(_get_color(config_name))

    # 绘制分组柱状图
    n_layers = len(all_variances[0]) if all_variances else 0
    n_configs = len(configs)
    bar_width = 0.25
    x = np.arange(n_layers)

    for ci in range(n_configs):
        offset = (ci - n_configs / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, all_variances[ci], bar_width,
                       label=labels[ci], color=colors_list[ci], alpha=0.85, edgecolor='white')

    ax1.set_xlabel('Network Layer')
    ax1.set_ylabel('Mean Gradient Variance')
    ax1.set_title(f'(a) Per-Layer Gradient Variance ({dataset_name})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names[:n_layers] if layer_names else range(n_layers),
                         rotation=45, ha='right')
    ax1.legend(frameon=True)
    ax1.set_yscale('log')

    # 子图(b): 各层梯度范数对比
    ax2 = fig.add_subplot(2, 3, 3)
    for ci, config_name in enumerate(configs):
        seed_h = histories[config_name][0]
        grad_stats = seed_h.get('gradient_stats', {})
        layer_names = seed_h.get('layer_names', [])
        norms = []
        for name in layer_names:
            if name in grad_stats:
                norms.append(grad_stats[name]['mean_norm'])
            else:
                norms.append(0)
        ax2.plot(range(len(norms)), norms, 'o-', color=_get_color(config_name),
                 label=_label_to_english(config_name), linewidth=1.5, markersize=6)

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Gradient Norm')
    ax2.set_title(f'(b) Per-Layer Gradient Norm')
    ax2.legend(frameon=True)
    ax2.set_yscale('log')

    # 子图(c): 梯度方差随epoch变化 (取第一层代表)
    ax3 = fig.add_subplot(2, 3, 4)
    for config_name in configs:
        seed_h = histories[config_name][0]
        grad_stats = seed_h.get('gradient_stats', {})
        layer_names = seed_h.get('layer_names', [])

        # 取中间层或第一层的方差曲线
        if layer_names:
            target_layer = layer_names[min(1, len(layer_names) - 1)]
            if target_layer in grad_stats:
                var_curve = grad_stats[target_layer]['var_curve']
                epochs = np.arange(1, len(var_curve) + 1)
                ax3.plot(epochs, var_curve, color=_get_color(config_name),
                         label=_label_to_english(config_name), linewidth=1.5)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Variance')
    ax3.set_title(f'(c) Gradient Variance Evolution (Layer 2)')
    ax3.legend(frameon=True)

    # 子图(d): 最大/最小梯度范数比随epoch变化
    ax4 = fig.add_subplot(2, 3, 5)
    for config_name in configs:
        seed_h = histories[config_name][0]
        grad_stats = seed_h.get('gradient_stats', {})
        layer_names = seed_h.get('layer_names', [])

        if layer_names:
            # 计算每个epoch的max/min比
            n_epochs = len(seed_h['train_loss'])
            ratios = []
            for epoch_idx in range(n_epochs):
                epoch_norms = []
                for name in layer_names:
                    if name in grad_stats and epoch_idx < len(grad_stats[name]['norm_curve']):
                        epoch_norms.append(grad_stats[name]['norm_curve'][epoch_idx])
                if len(epoch_norms) >= 2:
                    min_n = min(epoch_norms)
                    ratios.append(max(epoch_norms) / (min_n + 1e-8) if min_n > 1e-8 else max(epoch_norms) / 1e-8)
                else:
                    ratios.append(1.0)
            ax4.plot(range(1, len(ratios) + 1), ratios, color=_get_color(config_name),
                     label=_label_to_english(config_name), linewidth=1.5)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Max/Min Gradient Norm Ratio')
    ax4.set_title(f'(d) Gradient Norm Ratio Across Layers')
    ax4.legend(frameon=True)
    # 添加y=1参考线
    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    # 子图(e): 梯度SNR对比
    ax5 = fig.add_subplot(2, 3, 6)
    for config_name in configs:
        seed_h = histories[config_name][0]
        grad_stats = seed_h.get('gradient_stats', {})
        layer_names = seed_h.get('layer_names', [])

        snrs = []
        for name in layer_names:
            if name in grad_stats:
                mean_g = abs(grad_stats[name]['mean_gradient'])
                std_g = grad_stats[name]['std_gradient']
                snrs.append(mean_g / (std_g + 1e-8))
            else:
                snrs.append(0)

        ax5.plot(range(len(snrs)), snrs, 's-', color=_get_color(config_name),
                 label=_label_to_english(config_name), linewidth=1.5, markersize=5)

    ax5.set_xlabel('Layer Index')
    ax5.set_ylabel('Gradient SNR (|mean|/std)')
    ax5.set_title(f'(e) Gradient Signal-to-Noise Ratio')
    ax5.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig3_gradient_stability_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig3_gradient_stability_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig3: Gradient stability analysis saved for {dataset_name}")


def plot_mediation_results(mediation_result: Dict, dataset_name: str = 'MNIST'):
    """
    图4: 中介效应分析可视化

    (a) 路径系数图
    (b) Bootstrap分布
    """
    ensure_output_dir()
    bk = mediation_result.get('baron_kenny', {})
    boot = mediation_result.get('bootstrap', {})

    fig = plt.figure(figsize=(14, 5))

    # 子图(a): 路径系数图
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    ax1.set_title(f'(a) Mediation Path Diagram ({dataset_name})', fontsize=13, fontweight='bold')

    # 绘制矩形节点
    def draw_node(ax, x, y, text, color='#ECF0F1', fontsize=11):
        rect = FancyBboxPatch((x - 1.2, y - 0.3), 2.4, 0.6,
                               boxstyle="round,pad=0.1", facecolor=color,
                               edgecolor='#2C3E50', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # 绘制箭头
    def draw_arrow(ax, x1, y1, x2, y2, label='', color='#2C3E50', lw=2, style='-'):
        ax.annotate('', xy=(x2 - 0.15, y2 - 0.3 if y2 < y1 else y2 + 0.3),
                    xytext=(x1 + 0.15, y1 - 0.3 if y1 > y2 else y1 + 0.3),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=style))
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y + 0.15, label, fontsize=9, color=color, fontweight='bold')

    # A节点 (BatchNorm)
    draw_node(ax1, 2, 2, 'Batch Normalization', '#3498DB')
    # B节点 (Gradient Stability)
    draw_node(ax1, 5, 2, 'Gradient Stability', '#E67E22')
    # C节点 (Convergence)
    draw_node(ax1, 8, 2, 'Convergence Speed', '#2ECC71')

    # 路径
    a_val = bk.get('path_a', 0)
    b_val = bk.get('path_b', 0)
    c_val = bk.get('total_effect_c', 0)
    c_prime_val = bk.get('direct_effect_c_prime', 0)
    ab_val = bk.get('indirect_effect_ab', 0)

    a_sig = '***' if bk.get('path_a_p', 1) < 0.001 else '**' if bk.get('path_a_p', 1) < 0.01 else '*' if bk.get('path_a_p', 1) < 0.05 else ''
    b_sig = '***' if bk.get('path_b_p', 1) < 0.001 else '**' if bk.get('path_b_p', 1) < 0.01 else '*' if bk.get('path_b_p', 1) < 0.05 else ''

    # a路径: A → B
    draw_arrow(ax1, 2, 2.3, 4.85, 2.3, f'a = {a_val:.3f}{a_sig}', '#E74C3C', 2.5)
    # b路径: B → C
    draw_arrow(ax1, 5.15, 2.3, 7.85, 2.3, f'b = {b_val:.3f}{b_sig}', '#E74C3C', 2.5)
    # c'路径: A → C (直接)
    draw_arrow(ax1, 2, 1.5, 7.85, 1.5, f"c' = {c_prime_val:.3f}", '#7F8C8D', 1.5, '--')
    # c路径(总效应): 在下方
    draw_arrow(ax1, 2, 0.8, 7.85, 0.8, f'c = {c_val:.3f}', '#2980B9', 2, '-')

    # 间接效应标注
    ax1.text(5, 0.25, f'Indirect Effect (a*b) = {ab_val:.3f}\nMediation Ratio = {bk.get("mediation_ratio", 0):.1%}',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.7))

    # 子图(b): Bootstrap间接效应分布
    ax2 = fig.add_subplot(1, 2, 2)
    if 'indirect_effect_mean' in boot:
        mean_eff = boot['indirect_effect_mean']
        std_eff = boot['indirect_effect_std']
        ci_lower = boot.get('ci_lower', 0)
        ci_upper = boot.get('ci_upper', 0)
        sig_text = 'SIGNIFICANT' if boot.get('significant', False) else 'NOT SIGNIFICANT'
        sig_color = '#27AE60' if boot.get('significant', False) else '#E74C3C'

        if std_eff < 1e-15:
            ax2.axvline(x=mean_eff, color=COLORS['nobn'], linewidth=2,
                        label=f'Mean = {mean_eff:.4f}')
            ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax2.legend(frameon=True, fontsize=8)
            ax2.set_xlabel('Indirect Effect (a*b)')
        else:
            x = np.linspace(mean_eff - 4 * std_eff, mean_eff + 4 * std_eff, 200)
            y = stats_norm_pdf(x, mean_eff, std_eff)

            ax2.fill_between(x, y, alpha=0.3, color=COLORS['primary'])
            ax2.plot(x, y, color=COLORS['primary'], linewidth=2)

            ax2.axvline(x=ci_lower, color=COLORS['highlight'], linestyle='--', linewidth=1.5,
                        label=f'{boot["ci_level"]*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
            ax2.axvline(x=ci_upper, color=COLORS['highlight'], linestyle='--', linewidth=1.5)
            ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax2.axvline(x=mean_eff, color=COLORS['nobn'], linewidth=2,
                        label=f'Mean = {mean_eff:.4f}')

            ci_x = x[(x >= ci_lower) & (x <= ci_upper)]
            ci_y = stats_norm_pdf(ci_x, mean_eff, std_eff)
            ax2.fill_between(ci_x, ci_y, alpha=0.2, color=COLORS['highlight'])

            ax2.set_xlabel('Indirect Effect (a*b)')
            ax2.set_ylabel('Density')
            ax2.legend(frameon=True, fontsize=8)

        ax2.set_title(f'(b) Bootstrap Distribution (n={boot.get("n_bootstrap", 5000)})')
        ax2.text(0.95, 0.95, sig_text, transform=ax2.transAxes,
                 ha='right', va='top', fontsize=12, fontweight='bold',
                 color=sig_color,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Bootstrap results\nnot available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('(b) Bootstrap Distribution')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig4_mediation_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig4_mediation_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig4: Mediation analysis saved for {dataset_name}")


def stats_norm_pdf(x, mean, std):
    """正态分布PDF (避免scipy依赖冗余)"""
    std = max(std, 1e-15)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def plot_convergence_comparison(histories: Dict[str, List[Dict]], dataset_name: str = 'MNIST'):
    """
    图5: 收敛速度综合对比

    (a) 达到目标准确率所需epoch数
    (b) 最终准确率对比
    (c) 训练时间对比
    """
    ensure_output_dir()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    configs = list(histories.keys())
    x_labels = [_label_to_english(c) for c in configs]
    colors = [_get_color(c) for c in configs]
    x = np.arange(len(configs))

    # (a) 收敛epoch数
    convergence_epochs = []
    convergence_stds = []
    for config_name in configs:
        epochs_list = [h['convergence_epoch'] for h in histories[config_name]]
        convergence_epochs.append(np.mean(epochs_list))
        convergence_stds.append(np.std(epochs_list))

    bars1 = axes[0].bar(x, convergence_epochs, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    axes[0].errorbar(x, convergence_epochs, yerr=convergence_stds, fmt='none',
                     ecolor='black', capsize=5, capthick=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[0].set_ylabel('Epochs to Converge')
    axes[0].set_title('(a) Convergence Speed')
    # 添加数值标签
    for bar, val in zip(bars1, convergence_epochs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (b) 最终准确率
    best_accs = []
    best_acc_stds = []
    for config_name in configs:
        acc_list = [h['best_acc'] for h in histories[config_name]]
        best_accs.append(np.mean(acc_list))
        best_acc_stds.append(np.std(acc_list))

    bars2 = axes[1].bar(x, best_accs, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    axes[1].errorbar(x, best_accs, yerr=best_acc_stds, fmt='none',
                     ecolor='black', capsize=5, capthick=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[1].set_ylabel('Best Test Accuracy')
    axes[1].set_title('(b) Final Accuracy')
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    for bar, val in zip(bars2, best_accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (c) 训练时间
    train_times = []
    time_stds = []
    for config_name in configs:
        time_list = [sum(h['epoch_times']) for h in histories[config_name]]
        train_times.append(np.mean(time_list))
        time_stds.append(np.std(time_list))

    bars3 = axes[2].bar(x, train_times, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    axes[2].errorbar(x, train_times, yerr=time_stds, fmt='none',
                     ecolor='black', capsize=5, capthick=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, rotation=15, ha='right')
    axes[2].set_ylabel('Total Training Time (s)')
    axes[2].set_title('(c) Training Time')
    for bar, val in zip(bars3, train_times):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig5_convergence_comparison_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig5_convergence_comparison_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig5: Convergence comparison saved for {dataset_name}")


def plot_gradient_distribution_heatmap(histories: Dict[str, List[Dict]], dataset_name: str = 'MNIST'):
    """
    图6: 梯度分布热力图

    展示各层在不同epoch的梯度方差变化
    """
    ensure_output_dir()
    configs = list(histories.keys())
    n_configs = len(configs)

    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 4.5))
    if n_configs == 1:
        axes = [axes]

    for ci, config_name in enumerate(configs):
        seed_h = histories[config_name][0]
        grad_stats = seed_h.get('gradient_stats', {})
        layer_names = seed_h.get('layer_names', [])
        n_epochs = len(seed_h['train_loss'])

        # 构建热力图矩阵
        n_layers = len([n for n in layer_names if n in grad_stats])
        heatmap_data = np.zeros((n_layers, n_epochs))
        valid_names = [n for n in layer_names if n in grad_stats]

        for li, name in enumerate(valid_names):
            var_curve = grad_stats[name]['var_curve']
            heatmap_data[li, :len(var_curve)] = var_curve

        # log scale for better visualization
        heatmap_data_log = np.log10(heatmap_data + 1e-15)

        im = axes[ci].imshow(heatmap_data_log, aspect='auto', cmap='YlOrRd',
                              interpolation='bilinear')
        axes[ci].set_xlabel('Epoch')
        axes[ci].set_ylabel('Layer')
        axes[ci].set_title(f'{_label_to_english(config_name)}')
        axes[ci].set_yticks(range(n_layers))
        axes[ci].set_yticklabels(valid_names, fontsize=8)

        # 颜色条
        cbar = plt.colorbar(im, ax=axes[ci], shrink=0.8)
        cbar.set_label('log10(Variance)', fontsize=8)

    plt.suptitle(f'Gradient Variance Evolution ({dataset_name})', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig6_gradient_heatmap_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig6_gradient_heatmap_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig6: Gradient heatmap saved for {dataset_name}")


def plot_stability_vs_convergence(mediation_data: Dict, dataset_name: str = 'MNIST'):
    """
    图7: 梯度稳定性 vs 收敛速度散点图

    直观展示B-C关系的证据
    """
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 从mediation_data中提取
    # mediation_data的结构: {'all_results': [...], ...}
    all_results = mediation_data.get('all_results', [])

    if len(all_results) > 0:
        stability_scores = np.array([r['gradient_stability'] for r in all_results])
        convergence_speeds = np.array([r['convergence_speed'] for r in all_results])
        bn_flags = np.array([r['use_bn'] for r in all_results])

        # 散点图
        colors = [COLORS['bn_pre'] if bn else COLORS['nobn'] for bn in bn_flags]
        markers = ['o' if bn else 's' for bn in bn_flags]

        for i in range(len(stability_scores)):
            axes[0].scatter(stability_scores[i], convergence_speeds[i],
                           c=colors[i], marker=markers[i], s=80, alpha=0.8,
                           edgecolors='white', linewidth=0.5)

        # 回归线
        z = np.polyfit(stability_scores, convergence_speeds, 1)
        p = np.poly1d(z)
        x_line = np.linspace(stability_scores.min(), stability_scores.max(), 100)
        axes[0].plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=1.5)

        # 计算相关系数
        corr = np.corrcoef(stability_scores, convergence_speeds)[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[0].set_xlabel('Gradient Stability Score')
        axes[0].set_ylabel('Convergence Speed (1/epochs)')
        axes[0].set_title(f'(a) Stability vs Convergence ({dataset_name})')

        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['bn_pre'],
                   markersize=10, label='With BN'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['nobn'],
                   markersize=10, label='Without BN'),
        ]
        axes[0].legend(handles=legend_elements, frameon=True, loc='lower right')

        # 子图(b): 分组对比 (箱线图)
        nobn_stability = stability_scores[~bn_flags.astype(bool)]
        bn_stability = stability_scores[bn_flags.astype(bool)]

        box_data = []
        box_labels = []
        box_colors = []
        if len(nobn_stability) > 0:
            box_data.append(nobn_stability)
            box_labels.append('w/o BN')
            box_colors.append(COLORS['nobn'])
        if len(bn_stability) > 0:
            box_data.append(bn_stability)
            box_labels.append('With BN')
            box_colors.append(COLORS['bn_pre'])

        if len(box_data) > 0:
            bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True,
                              widths=0.4,
                              medianprops={'color': 'black', 'linewidth': 2})
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # 叠加散点
            for i, data in enumerate(box_data):
                jitter = np.random.normal(0, 0.04, len(data))
                axes[1].scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                               alpha=0.6, color=box_colors[i], edgecolors='white')

        axes[1].set_ylabel('Gradient Stability Score')
        axes[1].set_title(f'(b) Group Comparison ({dataset_name})')

        # t检验
        if len(nobn_stability) > 1 and len(bn_stability) > 1:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(bn_stability, nobn_stability)
            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            axes[1].text(0.5, 0.95, f't-test: p = {p_val:.4f} {sig_marker}',
                        transform=axes[1].transAxes, ha='center', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig7_stability_vs_convergence_{dataset_name}.png'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig7_stability_vs_convergence_{dataset_name}.pdf'))
    plt.close()
    print(f"  [OK] Fig7: Stability vs Convergence saved for {dataset_name}")


def generate_all_figures(histories: Dict[str, List[Dict]],
                          mediation_result: Dict,
                          dataset_name: str = 'MNIST',
                          output_dir: str = None):
    """生成所有图表"""
    global OUTPUT_DIR
    if output_dir is not None:
        OUTPUT_DIR = os.path.join(output_dir, 'figures')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Generating figures for {dataset_name}")
    print(f"Output: {OUTPUT_DIR}")
    print('='*60)

    plot_loss_curves(histories, dataset_name)
    plot_accuracy_curves(histories, dataset_name)
    plot_gradient_stability(histories, dataset_name)
    plot_mediation_results(mediation_result, dataset_name)
    plot_convergence_comparison(histories, dataset_name)
    plot_gradient_distribution_heatmap(histories, dataset_name)
    plot_stability_vs_convergence(mediation_result, dataset_name)

    print(f"All figures saved to: {OUTPUT_DIR}")
