import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
import time
import os
import warnings
from scipy.stats import mode
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score


plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 提高图像分辨率


def load_data(filepath):
    """增强数据加载与验证"""
    data = pd.read_csv(filepath)

    # 特征提取与验证
    X = data.iloc[:, 3:303].astype(np.float64)
    if X.isnull().any().any():
        X = X.fillna(X.mean())  # 均值填充缺失值

    # 标签编码与验证
    y = data['label'].values.astype('float32')
    if len(np.unique(y)) < 2:
        raise ValueError("目标变量需要至少2个类别")

    meta_cols = data[['drug', 'disease', 'label']]
    disease_cols = [col for col in data.columns if col.endswith('_y')]

    return X, y, meta_cols, disease_cols


def safe_feature_selection(X, y, k):
    """更安全的特征选择实现"""
    # 确保输入为numpy数组
    X = np.asarray(X)
    y = np.asarray(y)

    # 1. 方差筛选
    var_selector = VarianceThreshold(threshold=0.01)
    try:
        X_high_var = var_selector.fit_transform(X)
        high_var_indices = var_selector.get_support(indices=True)
    except ValueError:
        print("方差筛选失败，使用原始特征")
        X_high_var = X
        high_var_indices = np.arange(X.shape[1])

    # 2. 安全计算F值
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f_scores, p_values = f_classif(X_high_var, y)
            f_scores = np.nan_to_num(f_scores, nan=0, posinf=0, neginf=0)
        except:
            print("F检验计算失败，使用随机森林特征重要性")
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_high_var, y)
            f_scores = rf.feature_importances_
            p_values = np.zeros_like(f_scores)  # 创建虚拟p值

    # 3. 选择top k特征
    valid_indices = np.where(f_scores > 0)[0]
    if len(valid_indices) == 0:
        print("警告：没有找到有效特征，使用前k个")
        valid_indices = np.arange(len(f_scores))

    k = min(k, len(valid_indices))
    top_k_indices = np.argsort(f_scores[valid_indices])[-k:][::-1]
    selected_indices = high_var_indices[valid_indices[top_k_indices]]

    # 构建完整F值数组
    full_f_scores = np.zeros(X.shape[1])
    full_f_scores[high_var_indices] = f_scores

    # 构建完整p值数组
    full_p_values = np.ones(X.shape[1])
    full_p_values[high_var_indices] = p_values

    return X[:, selected_indices], selected_indices, full_f_scores, full_p_values


def plot_combined_fscores(full_f_scores, k140_scores, k160_scores, output_dir):
    """三合一F值对比图（无统计表格版）"""
    plt.figure(figsize=(12, 6))

    # 计算统计量
    stats = {
        'full': {
            'mean': np.mean(full_f_scores[full_f_scores > 0]),
            'median': np.median(full_f_scores[full_f_scores > 0]),
            'max': np.max(full_f_scores)
        },
        'k140': {
            'mean': np.mean(k140_scores),
            'median': np.median(k140_scores),
            'max': np.max(k140_scores)
        },
        'k160': {
            'mean': np.mean(k160_scores),
            'median': np.median(k160_scores),
            'max': np.max(k160_scores)
        }
    }

    # 配置参数
    hist_kwargs = {
        'bins': 50,
        'alpha': 0.6,
        'edgecolor': 'black',
        'linewidth': 0.5,
        'density': False
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = [
        f'全特征 (n={len(full_f_scores)}, μ={stats["full"]["mean"]:.1f})',
        f'k=140 (n={len(k140_scores)}, μ={stats["k140"]["mean"]:.1f})',
        f'k=160 (n={len(k160_scores)}, μ={stats["k160"]["mean"]:.1f})'
    ]

    # 绘制堆叠直方图
    plt.hist(
        [full_f_scores[full_f_scores > 0], k140_scores, k160_scores],
        color=colors,
        label=labels,
        **hist_kwargs,
        stacked=False
    )

    # 图表装饰
    title = (
        "F值分布对比\n"
        f"全特征: 最大={stats['full']['max']:.1f} 中位数={stats['full']['median']:.1f}\n"
        f"k=140: 最大={stats['k140']['max']:.1f} 中位数={stats['k140']['median']:.1f}\n"
        f"k=160: 最大={stats['k160']['max']:.1f} 中位数={stats['k160']['median']:.1f}"
    )
    plt.title(title, pad=20, fontsize=12)
    plt.xlabel('F值', fontsize=10)
    plt.ylabel('频数 (对数尺度)', fontsize=10)
    plt.yscale('log')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_fscore_comparison.png'), dpi=300)
    plt.close()


def robust_evaluation(X, y, model):
    """鲁棒的交叉验证评估"""
    try:
        # 定义评分指标（使用make_scorer确保兼容性）
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }

        # 执行交叉验证
        cv_results = cross_validate(
            model, X, y, cv=5,
            scoring=scoring,
            return_train_score=False,
            error_score='raise'
        )

        # 计算训练时间
        start_time = time.time()
        model.fit(X, y)
        train_time = time.time() - start_time

        # 返回格式化结果
        return {
            'accuracy': np.mean(cv_results['test_accuracy']),
            'balanced_accuracy': np.mean(cv_results['test_balanced_accuracy']),
            'f1_macro': np.mean(cv_results['test_f1_macro']),
            'roc_auc': np.mean(cv_results['test_roc_auc']),
            'train_time': train_time
        }

    except Exception as e:
        print(f"评估失败: {str(e)}")
        return {
            'accuracy': np.nan,
            'balanced_accuracy': np.nan,
            'f1_macro': np.nan,
            'roc_auc': np.nan,
            'train_time': np.nan
        }


def save_feature_importance(f_scores, selected_indices, feature_names, output_dir, filename):
    """保存特征重要性数据"""
    feature_importance = pd.DataFrame({
        'feature_index': selected_indices,
        'feature_name': [feature_names[i] for i in selected_indices],
        'f_score': f_scores[selected_indices]
    }).sort_values('f_score', ascending=False)

    feature_importance.to_csv(os.path.join(output_dir, filename), index=False)


def plot_performance_comparison(results_df, output_dir):
    """优化间距的训练时间对比图"""
    plt.figure(figsize=(12, 6))  # 加宽图形

    # 创建带特征个数的x轴标签
    x_labels = [f"{int(k)}\n({int(results_df[results_df['k'] == k]['selected_features'].values[0])}个)"
                for k in results_df['k']]

    # 调整条形图位置和宽度
    bar_positions = np.arange(len(results_df)) * 1.5  # 增加间距系数
    bar_width = 0.5

    # 训练时间对比图
    ax = plt.gca()
    ax.bar(
        bar_positions,
        results_df['selected_train_time'],
        width=bar_width,
        color='lightblue',
        label='选中特征'
    )

    # 折线图（保持与条形图对齐）
    ax.plot(
        bar_positions,
        results_df['full_train_time'],
        color='red',
        marker='o',
        label='全特征'
    )

    # 设置x轴刻度
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(x_labels)

    # 添加数值标签
    for i, row in results_df.iterrows():
        ax.text(
            bar_positions[i],
            row['selected_train_time'] + 0.05,
            f"{row['selected_train_time']:.2f}s",
            ha='center',
            color='blue'
        )
        ax.text(
            bar_positions[i],
            row['full_train_time'] + 0.05,
            f"{row['full_train_time']:.2f}s",
            ha='center',
            color='red'
        )

    # 图表装饰
    plt.title('训练时间对比（全特征 vs 选中特征）')
    plt.xlabel('特征数量（实际选中数）')
    plt.ylabel('训练时间 (秒)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 调整x轴范围增加两侧留白
    plt.xlim(bar_positions[0] - 1, bar_positions[-1] + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison_spaced.png'))
    plt.close()


def generate_feature_selection_report(results_df, output_dir):
    """生成特征选择报告"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    report = f"""
====================
特征选择分析报告
====================
总特征数量: {int(results_df.iloc[0]['k'] / results_df.iloc[0]['feature_reduction_ratio'])}
分析的特征数量范围: {results_df['k'].min()} 到 {results_df['k'].max()}

最佳性能配置:
- 最高准确率: {results_df['selected_accuracy'].max():.3f} (k={results_df.loc[results_df['selected_accuracy'].idxmax()]['k']})
- 最高ROC AUC: {results_df['selected_roc_auc'].max():.3f} (k={results_df.loc[results_df['selected_roc_auc'].idxmax()]['k']})
- 最短训练时间: {results_df['selected_train_time'].min():.2f}秒 (k={results_df.loc[results_df['selected_train_time'].idxmin()]['k']})

平均改进:
- 训练时间减少: {(results_df['full_train_time'] - results_df['selected_train_time']).mean():.2f}秒
- 特征减少比例: {results_df['feature_reduction_ratio'].mean():.1%}

特征重要性统计:
- 平均最大F值: {results_df['f_score_max'].mean():.3f}
- 平均F值: {results_df['f_score_mean'].mean():.3f}
"""
    with open(os.path.join(output_dir, 'feature_selection_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

def main_pipeline():
    """主流程"""
    # 配置参数
    input_file = '/home/zhangcy/file/test3/数据/未归一化/undersample/KSU_汉明.csv'
    output_dir = '/home/zhangcy/file/test3/数据zhang/未归一化/after_feature_select/F检验有效性'
    k_values = [140, 160]  # 重点关注这两个特征数量

    # 加载数据
    try:
        X, y, meta_cols, disease_cols = load_data(input_file)
        feature_names = X.columns.tolist()
        print(f"数据加载成功，形状: {X.shape}, 类别数: {len(np.unique(y))}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 主流程
    results = []
    full_f_scores = None
    k140_scores = None
    k160_scores = None

    for k in k_values:
        print(f"\n{'=' * 30}\n处理 k={k}\n{'=' * 30}")
        try:
            # 特征选择
            X_selected, selected_indices, f_scores, _ = safe_feature_selection(X, y, k)
            print(f"实际选中有效特征: {len(selected_indices)}/{k}")

            # 保存特征重要性
            save_feature_importance(f_scores, selected_indices, feature_names,
                                    output_dir, f'feature_importance_k{k}.csv')

            # 记录F值数据用于后续对比
            if k == 140:
                k140_scores = f_scores[selected_indices]
            elif k == 160:
                k160_scores = f_scores[selected_indices]

            # 模型评估
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # 全特征评估（仅第一次执行）
            if full_f_scores is None:
                full_results = robust_evaluation(X, y, model)
                full_f_scores = f_scores  # 保存全特征F值

            # 选中特征评估
            selected_results = robust_evaluation(X_selected, y, model)

            # 保存结果
            result = {
                'k': k,
                'selected_features': len(selected_indices),
                'f_score_max': np.max(f_scores[selected_indices]),
                'f_score_mean': np.mean(f_scores[selected_indices]),
                'full_accuracy': full_results['accuracy'],
                'selected_accuracy': selected_results['accuracy'],
                'full_roc_auc': full_results['roc_auc'],
                'selected_roc_auc': selected_results['roc_auc'],
                'f1_macro_diff': full_results['f1_macro'] - selected_results['f1_macro'],
                'full_train_time': full_results['train_time'],
                'selected_train_time': selected_results['train_time'],
                'feature_reduction_ratio': 1 - len(selected_indices) / X.shape[1]
            }
            results.append(result)

            # 打印简要报告
            print(f"\n评估结果 (k={k}):")
            print(f"全特征准确率: {full_results['accuracy']:.3f}")
            print(f"选中特征准确率: {selected_results['accuracy']:.3f}")
            print(f"ROC AUC差异: {full_results['roc_auc'] - selected_results['roc_auc']:.3f}")
            print(f"训练时间减少: {full_results['train_time'] - selected_results['train_time']:.2f}秒")

        except Exception as e:
            print(f"处理k={k}时出错: {str(e)}")
            continue

    # 绘制三合一F值对比图
    if full_f_scores is not None and k140_scores is not None and k160_scores is not None:
        plot_combined_fscores(full_f_scores, k140_scores, k160_scores, output_dir)

    # 保存最终结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'final_results.csv'), index=False)

    # 绘制性能对比图
    plot_performance_comparison(results_df, output_dir)

    # 生成报告
    generate_feature_selection_report(results_df, output_dir)
    print("\n流程完成！结果已保存至:", output_dir)


if __name__ == "__main__":
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.3f}'.format)

    # 运行主流程
    main_pipeline()