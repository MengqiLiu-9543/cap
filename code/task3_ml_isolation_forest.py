#!/usr/bin/env python3
"""
任务3：真正的Isolation Forest机器学习实现
使用scikit-learn训练异常检测模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import json

print("=" * 80)
print("任务3：Isolation Forest机器学习异常检测")
print("=" * 80)
print()

# ============================================================================
# 1. 加载数据
# ============================================================================

print("正在加载数据...")
import os
# 获取脚本所在目录的父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_file = os.path.join(parent_dir, 'data', 'task3_oncology_drug_event_pairs.csv')
df = pd.read_csv(data_file)
print(f"✓ 加载了 {len(df)} 条记录\n")

# ============================================================================
# 2. 特征工程
# ============================================================================

print("正在进行特征工程...")

# 统计每个药物-事件对的特征
from collections import Counter, defaultdict

drug_event_features = defaultdict(lambda: {
    'count': 0,
    'serious_count': 0,
    'death_count': 0,
    'hosp_count': 0,
    'life_threat_count': 0,
    'disable_count': 0,
    'ages': [],
    'drug_counts': []
})

# 统计全局信息
drug_totals = Counter()
event_totals = Counter()
total_reports = len(df)

for _, row in df.iterrows():
    drug = row['target_drug']
    event = row['adverse_event']
    pair = f"{drug}||{event}"
    
    drug_totals[drug] += 1
    event_totals[event] += 1
    
    drug_event_features[pair]['count'] += 1
    
    # 统计严重性
    try:
        if str(row.get('is_serious', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['serious_count'] += 1
        if str(row.get('is_death', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['death_count'] += 1
        if str(row.get('is_hospitalization', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['hosp_count'] += 1
        if str(row.get('is_lifethreatening', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['life_threat_count'] += 1
        if str(row.get('is_disabling', '0')).strip() in ['1', '1.0']:
            drug_event_features[pair]['disable_count'] += 1
    except:
        pass
    
    # 收集年龄和药物数量
    try:
        age = float(row.get('patient_age', 0))
        if age > 0 and age < 120:
            drug_event_features[pair]['ages'].append(age)
    except:
        pass
    
    try:
        drug_count = int(row.get('drug_count', 0))
        if drug_count > 0:
            drug_event_features[pair]['drug_counts'].append(drug_count)
    except:
        pass

print(f"✓ 识别了 {len(drug_event_features)} 个唯一的药物-事件对\n")

# ============================================================================
# 3. 构建特征矩阵
# ============================================================================

print("正在构建特征矩阵...")

feature_data = []
pair_names = []

for pair, stats in drug_event_features.items():
    drug, event = pair.split('||')
    
    count = stats['count']
    drug_total = drug_totals[drug]
    event_total = event_totals[event]
    
    # 计算PRR
    a = count
    b = drug_total - count
    c = event_total - count
    d = total_reports - drug_total - event_total + count
    
    prr = 0
    if b > 0 and c > 0 and d > 0:
        prr = (a / b) / (c / d) if (c / d) > 0 else 0
    
    # 计算ROR
    ror = 0
    if b > 0 and c > 0 and d > 0:
        ror = (a * d) / (b * c) if (b * c) > 0 else 0
    
    # 计算卡方
    expected_a = (drug_total * event_total) / total_reports
    chi2 = 0
    if expected_a > 0:
        chi2 = ((a - expected_a) ** 2) / expected_a
    
    # 计算严重性比例
    serious_rate = stats['serious_count'] / count if count > 0 else 0
    death_rate = stats['death_count'] / count if count > 0 else 0
    hosp_rate = stats['hosp_count'] / count if count > 0 else 0
    life_threat_rate = stats['life_threat_count'] / count if count > 0 else 0
    disable_rate = stats['disable_count'] / count if count > 0 else 0
    
    # 计算频率
    report_freq = count / total_reports
    
    # 平均年龄
    avg_age = np.mean(stats['ages']) if stats['ages'] else 0
    
    # 平均药物数量
    avg_drug_count = np.mean(stats['drug_counts']) if stats['drug_counts'] else 0
    
    # 构建特征向量
    features = [
        count,                  # 报告数量
        prr,                    # PRR
        ror,                    # ROR
        chi2,                   # 卡方
        serious_rate,           # 严重率
        death_rate,             # 死亡率
        hosp_rate,              # 住院率
        life_threat_rate,       # 危及生命率
        disable_rate,           # 致残率
        report_freq,            # 报告频率
        np.log(count + 1),      # log频率
        avg_age,                # 平均年龄
        avg_drug_count          # 平均药物数
    ]
    
    feature_data.append(features)
    pair_names.append(pair)

X = np.array(feature_data)
print(f"✓ 特征矩阵形状: {X.shape}")
print(f"✓ 特征数量: {X.shape[1]}\n")

# ============================================================================
# 4. 数据标准化
# ============================================================================

print("正在标准化特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ 标准化完成\n")

# ============================================================================
# 5. 训练Isolation Forest模型
# ============================================================================

print("正在训练Isolation Forest模型...")

# 设置参数
contamination = 0.15  # 预期异常比例15%
random_state = 42

# 训练模型
iso_forest = IsolationForest(
    contamination=contamination,
    random_state=random_state,
    n_estimators=100,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    verbose=0
)

# 拟合并预测
predictions = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

print(f"✓ 模型训练完成")
print(f"✓ 使用了 {iso_forest.n_estimators} 棵决策树")
print(f"✓ 异常阈值: {contamination * 100}%\n")

# ============================================================================
# 6. 分析结果
# ============================================================================

print("正在分析结果...")

# -1表示异常，1表示正常
n_anomalies = np.sum(predictions == -1)
n_normal = np.sum(predictions == 1)

print(f"✓ 检测到异常: {n_anomalies} 个 ({n_anomalies/len(predictions)*100:.1f}%)")
print(f"✓ 正常样本: {n_normal} 个 ({n_normal/len(predictions)*100:.1f}%)\n")

# ============================================================================
# 7. 保存结果
# ============================================================================

print("正在保存结果...")

# 创建结果DataFrame
results_df = pd.DataFrame({
    'drug_event_pair': pair_names,
    'drug': [p.split('||')[0] for p in pair_names],
    'event': [p.split('||')[1] for p in pair_names],
    'is_anomaly': predictions == -1,
    'anomaly_score': -anomaly_scores,  # 负数越大越异常
    'count': X[:, 0],
    'prr': X[:, 1],
    'ror': X[:, 2],
    'chi2': X[:, 3],
    'serious_rate': X[:, 4],
    'death_rate': X[:, 5],
    'hosp_rate': X[:, 6],
    'life_threat_rate': X[:, 7],
    'disable_rate': X[:, 8]
})

# 按异常分数排序
results_df = results_df.sort_values('anomaly_score', ascending=False)

# 保存所有结果
output_file1 = os.path.join(parent_dir, 'data', 'task3_ml_isolation_forest_results.csv')
results_df.to_csv(output_file1, index=False)
print(f"✓ 完整结果已保存: {output_file1}")

# 只保存异常样本
anomalies_df = results_df[results_df['is_anomaly'] == True]
output_file2 = os.path.join(parent_dir, 'data', 'task3_ml_anomalies_only.csv')
anomalies_df.to_csv(output_file2, index=False)
print(f"✓ 异常样本已保存: {output_file2} ({len(anomalies_df)} 个)\n")

# ============================================================================
# 8. 展示Top异常
# ============================================================================

print("=" * 80)
print("Top 30 最显著的异常（基于Isolation Forest）")
print("=" * 80)
print()

for i, row in anomalies_df.head(30).iterrows():
    print(f"[{i+1}] 异常分数: {row['anomaly_score']:.4f}")
    print(f"    药物: {row['drug']}")
    print(f"    不良事件: {row['event']}")
    print(f"    报告数: {int(row['count'])}")
    print(f"    PRR: {row['prr']:.3f} | ROR: {row['ror']:.3f} | χ²: {row['chi2']:.2f}")
    print(f"    死亡率: {row['death_rate']*100:.1f}% | 住院率: {row['hosp_rate']*100:.1f}%")
    print()

# ============================================================================
# 9. 按药物汇总
# ============================================================================

print("=" * 80)
print("按药物汇总的异常统计 (Top 20)")
print("=" * 80)
print()

drug_anomaly_counts = anomalies_df['drug'].value_counts()

for i, (drug, count) in enumerate(drug_anomaly_counts.head(20).items(), 1):
    drug_anomalies = anomalies_df[anomalies_df['drug'] == drug]
    max_score = drug_anomalies['anomaly_score'].max()
    top_events = drug_anomalies.nlargest(3, 'anomaly_score')['event'].tolist()
    
    print(f"{i}. {drug}:")
    print(f"   异常信号数: {count}")
    print(f"   最高异常分数: {max_score:.4f}")
    print(f"   Top事件: {', '.join(top_events)}")
    print()

# ============================================================================
# 10. Epcoritamab分析
# ============================================================================

epc_anomalies = anomalies_df[anomalies_df['drug'] == 'Epcoritamab']

if len(epc_anomalies) > 0:
    print("=" * 80)
    print("Epcoritamab 异常信号分析")
    print("=" * 80)
    print()
    print(f"检测到 {len(epc_anomalies)} 个异常信号")
    print()
    print("Top 10 Epcoritamab异常事件:")
    for i, row in epc_anomalies.head(10).iterrows():
        print(f"  {i+1}. {row['event']}")
        print(f"     异常分数: {row['anomaly_score']:.4f}")
        print(f"     PRR: {row['prr']:.3f} | 死亡率: {row['death_rate']*100:.1f}%")

print()
print("=" * 80)
print("任务3完成！真正的Isolation Forest机器学习模型已训练完成。")
print("=" * 80)

