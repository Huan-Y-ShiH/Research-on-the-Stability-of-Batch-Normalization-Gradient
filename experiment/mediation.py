"""
中介效应分析模块
验证路径: A(BatchNorm) → B(梯度稳定性) → C(收敛速度)

采用Baron & Kenny四步法 + Bootstrap检验 + Sobel检验
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional


class MediationAnalysis:
    """
    中介效应分析

    A (BatchNorm, 二值变量) → B (梯度稳定性, 连续变量) → C (收敛速度, 连续变量)

    理论框架:
    - 总效应 (c): A → C 直接作用
    - 间接效应 (a*b): A → B → C 中介路径
    - 直接效应 (c'): 控制B后A → C的直接作用
    - 中介比例: a*b / c
    """

    def __init__(self):
        self.results = {}

    def _regression(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """执行线性回归并返回系数、标准误、p值、R²"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        n = len(y)
        k = X.shape[1]
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)
        # 正确计算标准误
        X_centered = X - X.mean(axis=0)
        XtX = X_centered.T @ X_centered
        try:
            XtX_inv = np.linalg.inv(XtX)
            se_array = np.sqrt(rss / (n - k - 1) * np.diag(XtX_inv))
        except np.linalg.LinAlgError:
            se_array = np.zeros(k)
        se = se_array[0]
        t_stat = model.coef_[0] / se if se > 1e-15 else 0.0
        df = max(1, n - k - 1)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        r2 = 1 - rss / max(1e-15, np.sum((y - y.mean()) ** 2))
        return model.coef_[0] if k == 1 else model.coef_[0], se, p_value, r2

    def baron_kenny(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                    labels: Optional[List[str]] = None) -> Dict:
        """
        Baron & Kenny四步法

        Step 1: A → C (总效应, 必须显著)
        Step 2: A → B (A必须影响中介变量)
        Step 3: B → C (控制A后, 中介变量必须影响结果)
        Step 4: A + B → C (直接效应应小于总效应)

        Args:
            A: 自变量 (0/1)
            B: 中介变量 (梯度稳定性)
            C: 因变量 (收敛速度指标)
            labels: 可选标签
        """
        n = len(A)

        # Step 1: 总效应 c: A → C
        c, se_c, p_c, r2_c = self._regression(A, C)

        # Step 2: a路径: A → B
        a, se_a, p_a, r2_a = self._regression(A, B)

        # Step 3: b路径 (控制A): B → C
        X_ab = np.column_stack([A, B])
        model_b = LinearRegression()
        model_b.fit(X_ab, C)
        b = model_b.coef_[1]
        c_prime = model_b.coef_[0]  # 直接效应

        # 计算b的标准误
        residuals = C - model_b.predict(X_ab)
        rss = np.sum(residuals ** 2)
        n_samples = len(C)
        try:
            XtX_inv_b = np.linalg.inv(X_ab.T @ X_ab)
            se_b = np.sqrt(max(0, rss / max(1, n_samples - 3)) * XtX_inv_b[1, 1])
        except np.linalg.LinAlgError:
            se_b = 1e-8
        t_b = b / se_b if se_b > 1e-15 else 0.0
        df_b = max(1, n_samples - 3)
        p_b = 2 * (1 - stats.t.cdf(abs(t_b), df_b))
        r2_ab = 1 - rss / max(1e-15, np.sum((C - C.mean()) ** 2))

        # 间接效应 a*b
        indirect_effect = a * b
        # Sobel检验
        se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
        sobel_z = indirect_effect / se_indirect if se_indirect > 0 else 0
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        # 中介比例
        mediation_ratio = indirect_effect / c if c != 0 else 0

        return {
            'total_effect_c': c,
            'total_effect_se': se_c,
            'total_effect_p': p_c,
            'total_effect_r2': r2_c,
            'path_a': a,
            'path_a_se': se_a,
            'path_a_p': p_a,
            'path_a_r2': r2_a,
            'path_b': b,
            'path_b_se': se_b,
            'path_b_p': p_b,
            'direct_effect_c_prime': c_prime,
            'indirect_effect_ab': indirect_effect,
            'indirect_effect_se': se_indirect,
            'sobel_z': sobel_z,
            'sobel_p': sobel_p,
            'mediation_ratio': mediation_ratio,
            'r2_full_model': r2_ab,
            'n_samples': n,
            'mediation_detected': (p_a < 0.05 and p_b < 0.05),
            'is_full_mediation': (p_c < 0.05 and c_prime / c < 0.3) if c != 0 else False,
            'is_partial_mediation': (p_c < 0.05 and p_a < 0.05 and p_b < 0.05 and abs(c_prime) < abs(c)),
        }

    def bootstrap_mediation(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                            n_bootstrap: int = 5000, alpha: float = 0.05) -> Dict:
        """
        Bootstrap法估计间接效应的置信区间

        这是更稳健的中介效应检验方法，不依赖正态性假设
        """
        n = len(A)
        indirect_effects = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # 有放回抽样
            idx = np.random.choice(n, size=n, replace=True)
            A_boot, B_boot, C_boot = A[idx], B[idx], C[idx]

            # 计算a路径
            a_boot, _, _, _ = self._regression(A_boot, B_boot)
            # 计算b路径 (控制A)
            model = LinearRegression()
            model.fit(np.column_stack([A_boot, B_boot]), C_boot)
            b_boot = model.coef_[1]
            indirect_effects[i] = a_boot * b_boot

        # 百分位法置信区间
        ci_lower = np.percentile(indirect_effects, 100 * alpha / 2)
        ci_upper = np.percentile(indirect_effects, 100 * (1 - alpha / 2))
        mean_effect = np.mean(indirect_effects)
        std_effect = np.std(indirect_effects)

        return {
            'indirect_effect_mean': mean_effect,
            'indirect_effect_std': std_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_level': 1 - alpha,
            'significant': (ci_lower * ci_upper > 0),  # CI不包含0则显著
            'n_bootstrap': n_bootstrap,
        }


def run_mediation_analysis(all_results: List[Dict]) -> Dict:
    """
    对多种子实验结果执行完整中介分析
    """
    n = len(all_results)

    # 构建变量，过滤NaN
    A = []
    B = []
    C = []
    for r in all_results:
        a_val = 1.0 if r['use_bn'] else 0.0
        b_val = r['gradient_stability']
        c_val = r['convergence_speed']
        if np.isnan(b_val) or np.isinf(b_val) or np.isnan(c_val) or np.isinf(c_val):
            continue
        A.append(a_val)
        B.append(b_val)
        C.append(c_val)

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    if len(A) < 4:
        return {
            'baron_kenny': {'mediation_detected': False, 'error': 'insufficient_data'},
            'bootstrap': {'significant': False, 'error': 'insufficient_data'},
            'effect_sizes': {},
            'n_experiments': len(A),
        }

    analyzer = MediationAnalysis()
    bk_results = analyzer.baron_kenny(A, B, C)
    bootstrap_results = analyzer.bootstrap_mediation(A, B, C, n_bootstrap=5000)
    effect_sizes = compute_effect_sizes(all_results)

    return {
        'baron_kenny': bk_results,
        'bootstrap': bootstrap_results,
        'effect_sizes': effect_sizes,
        'n_experiments': n,
    }


def compute_effect_sizes(all_results: List[Dict]) -> Dict:
    """
    计算Cohen's d效应量

    对比有BN vs 无BN在各项指标上的效应量
    """
    bn_group = [r for r in all_results if r['use_bn']]
    nobn_group = [r for r in all_results if not r['use_bn']]

    if len(bn_group) == 0 or len(nobn_group) == 0:
        return {}

    def cohens_d(group1, group2):
        """计算Cohen's d"""
        g1 = np.array(group1)
        g2 = np.array(group2)
        n1, n2 = len(g1), len(g2)
        pooled_std = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0
        return (g1.mean() - g2.mean()) / pooled_std

    effects = {
        'gradient_stability': cohens_d(
            [r['gradient_stability'] for r in bn_group if not np.isnan(r['gradient_stability']) and not np.isinf(r['gradient_stability'])],
            [r['gradient_stability'] for r in nobn_group if not np.isnan(r['gradient_stability']) and not np.isinf(r['gradient_stability'])]
        ),
        'convergence_speed': cohens_d(
            [r['convergence_speed'] for r in bn_group if not np.isnan(r['convergence_speed']) and not np.isinf(r['convergence_speed'])],
            [r['convergence_speed'] for r in nobn_group if not np.isnan(r['convergence_speed']) and not np.isinf(r['convergence_speed'])]
        ),
        'best_accuracy': cohens_d(
            [r['best_acc'] for r in bn_group],
            [r['best_acc'] for r in nobn_group]
        ),
    }

    return effects
