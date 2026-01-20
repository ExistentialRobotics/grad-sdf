"""
从保存的参数搜索结果中，使用不同的评分公式重新排序和评估参数
"""

import json
from pathlib import Path

# 设置路径
data_dir = Path("/home/qihao/workplace/grad-sdf/data/newercollege-lidar")
json_file = data_dir / "param_search_raw_results.json"

# 读取保存的结果
print(f"正在读取保存的结果...")
with open(json_file, "r", encoding="utf-8") as f:
    all_results = json.load(f)
print(f"✓ 读取了 {len(all_results)} 组参数结果")


# ============================================================
# 在这里定义你的评分公式
# ============================================================
def custom_score(result):
    """
    自定义评分函数

    可用的字段：
        - recall: 召回率（方法二找到了多少方法一过滤的点）
        - precision: 精确率（方法二过滤的点中有多少是正确的）
        - f1_score: F1分数
        - false_positive_rate: 误报率（方法二误过滤的点的比例）
        - avg_agreement: 平均一致性
        - method1_filter_rate: 方法1的过滤率
        - method2_filter_rate: 方法2的过滤率
        - both_filtered: 两种方法都过滤的点数
        - only_method1: 只被方法1过滤的点数
        - only_method2: 只被方法2过滤的点数
        - total_points: 总点数

    返回值：分数越高越好
    """
    # 示例1: 原始评分 (召回率 - 1.0 × 误报率)
    # return result["recall"] - 1.0 * result["false_positive_rate"]

    # 示例2: 更重视召回率，轻微惩罚误报
    # return result["recall"] - 0.5 * result["false_positive_rate"]

    # 示例3: 平衡召回率和精确率（使用F1）
    # return result["f1_score"]

    # 示例4: 高召回率优先，强力惩罚误报
    # return result["recall"] - 2.0 * result["false_positive_rate"]

    # 示例5: 只关心召回率
    # return result["recall"]

    # 示例6: 召回率达到阈值后，最小化误报
    # if result["recall"] >= 0.95:
    #     return 1.0 - result["false_positive_rate"]  # 最小化误报
    # else:
    #     return result["recall"]  # 先保证召回率

    # 示例7: 综合考虑，使用加权和
    # return 2.0 * result["recall"] + 1.0 * result["precision"] - 3.0 * result["false_positive_rate"]

    # 当前使用的评分公式（可以修改）
    return result["recall"] - 1.0 * result["false_positive_rate"]


# ============================================================
# 重新评分
# ============================================================
print("\n正在使用新的评分公式重新评估...")
for result in all_results:
    result["custom_score"] = custom_score(result)

# 按新评分排序
all_results.sort(key=lambda x: x["custom_score"], reverse=True)

# ============================================================
# 输出结果
# ============================================================
print("\n" + "=" * 115)
print("重新评分结果 - Top 20")
print("=" * 115)
print(
    f"{'排名':<6} {'距离阈值':<12} {'邻居数':<8} {'新评分':<12} "
    f"{'召回率':<12} {'精确率':<12} {'F1':<12} {'误报率':<12}"
)
print("-" * 115)

for rank, result in enumerate(all_results[:20], 1):
    print(
        f"{rank:<6} {result['distance_threshold']:<12.2f} {result['n_neighbors']:<8} "
        f"{result['custom_score']:<12.4f} "
        f"{result['recall']*100:>10.2f}%  {result['precision']*100:>10.2f}%  "
        f"{result['f1_score']*100:>10.2f}%  {result['false_positive_rate']*100:>10.2f}%"
    )

print("\n" + "=" * 115)
print("🎯 新评分下的最佳参数:")
print("=" * 115)
best_result = all_results[0]
print(f"distance_threshold_method2 = {best_result['distance_threshold']:.2f}")
print(f"n_neighbors = {best_result['n_neighbors']}")

print(f"\n📊 核心性能指标:")
print(f"  新评分: {best_result['custom_score']:.4f}")
print(f"  召回率 (Recall): {best_result['recall']*100:.2f}%  <- 方法二找到了方法一过滤点的比例")
print(f"  精确率 (Precision): {best_result['precision']*100:.2f}%  <- 方法二过滤的点中正确的比例")
print(f"  F1-Score: {best_result['f1_score']*100:.2f}%")
print(f"  误报率: {best_result['false_positive_rate']*100:.2f}%  <- 方法二误过滤的点的比例")
print(f"  一致性: {best_result['avg_agreement']*100:.2f}%")

print(f"\n📈 详细统计:")
print(f"  总点数: {best_result['total_points']}")
print(f"  方法1过滤点数: {best_result['method1_filtered_total']} ({best_result['method1_filter_rate']*100:.2f}%)")
print(f"  方法2过滤点数: {best_result['method2_filtered_total']} ({best_result['method2_filter_rate']*100:.2f}%)")
print(
    f"\n  ✅ 两种方法都过滤 (正确): {best_result['both_filtered']} ({best_result['both_filtered']/best_result['total_points']*100:.2f}%)"
)
print(
    f"  ⚠️  只被方法1过滤 (漏检): {best_result['only_method1']} ({best_result['only_method1']/best_result['total_points']*100:.2f}%)"
)
print(
    f"  ❌ 只被方法2过滤 (误报): {best_result['only_method2']} ({best_result['only_method2']/best_result['total_points']*100:.2f}%)"
)

print(f"\n💡 解读:")
if best_result["recall"] > 0.95:
    print(f"  ✓ 召回率很高 ({best_result['recall']*100:.1f}%)，方法二能找到几乎所有方法一过滤的点")
elif best_result["recall"] > 0.85:
    print(f"  ✓ 召回率较高 ({best_result['recall']*100:.1f}%)，方法二能找到大部分方法一过滤的点")
else:
    print(f"  ✗ 召回率偏低 ({best_result['recall']*100:.1f}%)，方法二漏检了较多点")

if best_result["false_positive_rate"] < 0.01:
    print(f"  ✓ 误报率很低 ({best_result['false_positive_rate']*100:.1f}%)，方法二很少误过滤")
elif best_result["false_positive_rate"] < 0.05:
    print(f"  ✓ 误报率较低 ({best_result['false_positive_rate']*100:.1f}%)，方法二误过滤较少")
else:
    print(f"  ✗ 误报率偏高 ({best_result['false_positive_rate']*100:.1f}%)，方法二过滤了较多不该过滤的点")

# ============================================================
# 对比不同的评分策略
# ============================================================
print("\n" + "=" * 115)
print("📊 不同评分策略对比 (Top 3):")
print("=" * 115)

scoring_strategies = [
    ("只看召回率", lambda r: r["recall"]),
    ("只看F1", lambda r: r["f1_score"]),
    ("召回-0.5×误报", lambda r: r["recall"] - 0.5 * r["false_positive_rate"]),
    ("召回-1.0×误报", lambda r: r["recall"] - 1.0 * r["false_positive_rate"]),
    ("召回-2.0×误报", lambda r: r["recall"] - 2.0 * r["false_positive_rate"]),
]

for strategy_name, score_func in scoring_strategies:
    # 重新评分
    for result in all_results:
        result["temp_score"] = score_func(result)
    all_results.sort(key=lambda x: x["temp_score"], reverse=True)

    print(f"\n策略: {strategy_name}")
    print("-" * 115)
    for rank, result in enumerate(all_results[:3], 1):
        print(
            f"  {rank}. 阈值={result['distance_threshold']:.2f}, 邻居={result['n_neighbors']}, "
            f"召回率={result['recall']*100:.1f}%, 误报率={result['false_positive_rate']*100:.1f}%, "
            f"F1={result['f1_score']*100:.1f}%"
        )

print("\n" + "=" * 115)
