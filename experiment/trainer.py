"""
训练器：带梯度监控的训练循环
核心功能:
1. 注册梯度Hook记录每层梯度统计量
2. 同时记录损失/准确率收敛曲线
3. 支持多随机种子运行
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import time
import copy


class GradientMonitor:
    """梯度监控器 - 通过hook记录每层梯度的统计量"""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.handles = []
        # 每个epoch的梯度统计: {layer_name: {'mean': [], 'std': [], 'norm': [], 'var': []}}
        self.gradient_stats = defaultdict(lambda: defaultdict(list))
        self._current_epoch_stats = defaultdict(lambda: defaultdict(list))

    def _hook_fn(self, layer_name: str):
        def hook(grad):
            if grad is not None:
                g = grad.detach().cpu()
                m = g.mean().item()
                s = g.std().item()
                n = g.norm().item()
                v = g.var().item()
                # 丢弃NaN/Inf梯度（混合精度下偶发）
                if any(np.isnan(x) or np.isinf(x) for x in [m, s, n, v]):
                    return
                self._current_epoch_stats[layer_name]['mean'].append(m)
                self._current_epoch_stats[layer_name]['std'].append(s)
                self._current_epoch_stats[layer_name]['norm'].append(n)
                self._current_epoch_stats[layer_name]['var'].append(v)
        return hook

    def register(self):
        """注册hook到所有层的权重参数"""
        if hasattr(self.model, 'get_linear_conv_layers'):
            layers = self.model.get_linear_conv_layers()
        else:
            layers = self.model.get_linear_layers()
        for i, layer in enumerate(layers):
            if i < len(self.layer_names):
                h = layer.weight.register_hook(self._hook_fn(self.layer_names[i]))
                self.handles.append(h)

    def epoch_finalize(self):
        """每个epoch结束后，汇总本epoch的梯度统计"""
        for name in self.layer_names:
            if name in self._current_epoch_stats:
                for stat_name in ['mean', 'std', 'norm', 'var']:
                    vals = self._current_epoch_stats[name][stat_name]
                    if len(vals) > 0:
                        avg_val = float(np.mean(vals))
                        if np.isnan(avg_val) or np.isinf(avg_val):
                            avg_val = 0.0
                        self.gradient_stats[name][stat_name].append(avg_val)
                    else:
                        self.gradient_stats[name][stat_name].append(0.0)
        self._current_epoch_stats.clear()

    def remove(self):
        """移除所有hook"""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_summary(self) -> Dict:
        """获取梯度统计汇总"""
        summary = {}
        for name in self.layer_names:
            if name in self.gradient_stats:
                summary[name] = {
                    'mean_gradient': np.mean(self.gradient_stats[name]['mean']),
                    'std_gradient': np.mean(self.gradient_stats[name]['std']),
                    'mean_norm': np.mean(self.gradient_stats[name]['norm']),
                    'mean_var': np.mean(self.gradient_stats[name]['var']),
                    'var_curve': np.array(self.gradient_stats[name]['var']),
                    'norm_curve': np.array(self.gradient_stats[name]['norm']),
                }
        return summary


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device,
                grad_monitor: Optional[GradientMonitor] = None) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == 'cuda'  # GPU时启用混合精度加速
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                config: Dict, device: torch.device,
                verbose: bool = True) -> Dict:
    """
    完整训练流程，带梯度监控

    Args:
        model: 待训练模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        config: 训练配置
        device: 计算设备
        verbose: 是否打印进度

    Returns:
        包含所有训练记录和梯度统计的字典
    """
    epochs = config.get('epochs', 50)
    lr = config.get('lr', 0.01)
    seed = config.get('seed', 42)

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 梯度监控
    layer_names = model.get_layer_names()
    grad_monitor = GradientMonitor(model, layer_names)
    grad_monitor.register()

    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'gradient_stats': None,
        'epoch_times': [],
        'config': config,
        'layer_names': layer_names,
    }

    best_acc = 0.0
    best_model_state = None
    final_best_acc = 0.0  # 训练结束时的最终最佳准确率

    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_monitor
        )

        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(time.time() - epoch_start)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())

        # 梯度统计
        grad_monitor.epoch_finalize()

        # 学习率调度
        scheduler.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    final_best_acc = best_acc

    # 后处理：检测收敛epoch (第一个达到最终最佳准确率90%的epoch)
    convergence_epoch = epochs
    threshold = 0.9 * final_best_acc
    for epoch in range(epochs):
        if history['test_acc'][epoch] >= threshold:
            convergence_epoch = epoch + 1
            break

    # 移除hook
    grad_monitor.remove()
    history['gradient_stats'] = grad_monitor.get_summary()
    history['best_acc'] = best_acc
    history['best_model_state'] = best_model_state
    history['convergence_epoch'] = convergence_epoch

    # 恢复最佳模型
    model.load_state_dict(best_model_state)

    return history


def compute_gradient_stability_metrics(gradient_stats: Dict) -> Dict:
    """从梯度统计中计算梯度稳定性综合指标，含NaN保护"""
    layer_name_order = list(gradient_stats.keys())
    layer_norms = []
    temporal_variances = []

    for name in layer_name_order:
        stats = gradient_stats[name]
        norm = stats['mean_norm']
        if np.isnan(norm) or np.isinf(norm):
            norm = 0.0
        layer_norms.append(norm)

        var_curve = stats['var_curve']
        var_curve_clean = var_curve[~np.isnan(var_curve)] if len(var_curve) > 0 else np.array([0])
        var_curve_clean = var_curve_clean[~np.isinf(var_curve_clean)]
        if len(var_curve_clean) > 1:
            temporal_variances.append(float(np.std(var_curve_clean)))
        elif len(var_curve_clean) == 1:
            temporal_variances.append(0.0)
        else:
            temporal_variances.append(0.0)

    layer_norms_arr = np.array(layer_norms)
    mean_n = np.mean(layer_norms_arr)
    std_n = np.std(layer_norms_arr)
    cv_norms = std_n / (mean_n + 1e-8) if mean_n > 0 else 0.0

    avg_temporal_var = float(np.mean(temporal_variances)) if temporal_variances else 0.0

    max_n = max(layer_norms_arr)
    min_n = min(layer_norms_arr)
    max_min_ratio = max_n / (min_n + 1e-8) if min_n > 1e-8 else 1.0

    denom = cv_norms * avg_temporal_var + 1e-8
    stability_score = 1.0 / denom if denom > 1e-15 else 1e8

    if np.isnan(stability_score) or np.isinf(stability_score):
        stability_score = 1e8

    return {
        'layer_variance': float(cv_norms),
        'cross_layer_ratio': float(max_min_ratio),
        'temporal_variance': float(avg_temporal_var),
        'stability_score': float(stability_score),
        'layer_names': layer_name_order,
    }
