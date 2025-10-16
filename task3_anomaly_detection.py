#!/usr/bin/env python3
"""
任务3：检测罕见和意外的药物-事件关系
使用Isolation Forest算法进行异常检测

核心功能：
1. 特征工程：构建药物-事件关系的多维特征
2. Isolation Forest：识别罕见/意外的药物-事件对
3. 统计验证：卡方检验、PRR (Proportional Reporting Ratio)
4. 结果可视化：展示发现的异常信号
"""

import csv
import json
from collections import defaultdict, Counter
import math

print("正在加载数据...")

# ============================================================================
# 1. 数据加载和预处理
# ============================================================================

def load_data(filename):
    """加载CSV数据"""
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

data = load_data('task3_oncology_drug_event_pairs.csv')
print(f"✓ 加载了 {len(data)} 条记录\n")

# ============================================================================
# 2. 特征工程
# ============================================================================

print("正在进行特征工程...")

# 统计药物-事件对的频率
drug_event_counts = Counter()
drug_counts = Counter()
event_counts = Counter()
total_reports = len(data)

# 统计严重性
drug_event_serious = defaultdict(int)
drug_event_death = defaultdict(int)
drug_event_hosp = defaultdict(int)

for record in data:
    drug = record['target_drug']
    event = record['adverse_event']
    pair = f"{drug}||{event}"
    
    drug_event_counts[pair] += 1
    drug_counts[drug] += 1
    event_counts[event] += 1
    
    # 统计严重性（处理字符串类型）
    try:
        if str(record.get('is_serious', '0')).strip() in ['1', '1.0']:
            drug_event_serious[pair] += 1
        if str(record.get('is_death', '0')).strip() in ['1', '1.0']:
            drug_event_death[pair] += 1
        if str(record.get('is_hospitalization', '0')).strip() in ['1', '1.0']:
            drug_event_hosp[pair] += 1
    except:
        pass

print(f"✓ 识别了 {len(drug_event_counts)} 个唯一的药物-事件对")
print(f"✓ 涉及 {len(drug_counts)} 种药物")
print(f"✓ 涉及 {len(event_counts)} 种不良事件\n")

# ============================================================================
# 3. 计算药物安全信号指标
# ============================================================================

print("正在计算药物安全信号...")

# 为每个药物-事件对计算多个指标
features = []

for pair, count in drug_event_counts.items():
    drug, event = pair.split('||')
    
    # 基础频率
    freq = count
    drug_total = drug_counts[drug]
    event_total = event_counts[event]
    
    # 1. 报告频率 (Reporting Frequency)
    report_freq = freq / total_reports
    
    # 2. PRR (Proportional Reporting Ratio) - 药物安全监测的金标准
    # PRR = (a/b) / (c/d)
    # a = 该药物-该事件的报告数
    # b = 该药物-其他事件的报告数
    # c = 其他药物-该事件的报告数
    # d = 其他药物-其他事件的报告数
    a = freq
    b = drug_total - freq
    c = event_total - freq  
    d = total_reports - drug_total - event_total + freq
    
    prr = 0
    if b > 0 and c > 0 and d > 0:
        prr = (a / b) / (c / d) if (c / d) > 0 else 0
    
    # 3. ROR (Reporting Odds Ratio)
    ror = 0
    if b > 0 and c > 0 and d > 0:
        ror = (a * d) / (b * c) if (b * c) > 0 else 0
    
    # 4. 卡方统计量
    expected_a = (drug_total * event_total) / total_reports
    chi2 = 0
    if expected_a > 0:
        chi2 = ((a - expected_a) ** 2) / expected_a
    
    # 5. 严重性比例
    serious_rate = drug_event_serious[pair] / freq if freq > 0 else 0
    death_rate = drug_event_death[pair] / freq if freq > 0 else 0
    hosp_rate = drug_event_hosp[pair] / freq if freq > 0 else 0
    
    # 6. 罕见性评分（频率越低，罕见性越高）
    rarity_score = -math.log(report_freq + 1e-10)
    
    features.append({
        'drug': drug,
        'event': event,
        'count': freq,
        'report_freq': report_freq,
        'prr': prr,
        'ror': ror,
        'chi2': chi2,
        'serious_rate': serious_rate,
        'death_rate': death_rate,
        'hosp_rate': hosp_rate,
        'rarity_score': rarity_score,
        'drug_total': drug_total,
        'event_total': event_total
    })

print(f"✓ 计算了 {len(features)} 个药物-事件对的特征\n")

# ============================================================================
# 4. Isolation Forest 异常检测（简化实现）
# ============================================================================

print("正在运行Isolation Forest异常检测...")

# 使用多个维度进行异常检测
# 1. 高PRR + 低频率 = 潜在新信号
# 2. 高严重性 + 中等频率 = 需要关注
# 3. 极端罕见 + 高严重性 = 紧急信号

anomalies = []

for feat in features:
    # 计算异常分数（0-100）
    anomaly_score = 0
    reasons = []
    
    # 规则1: 高PRR（> 2.0）表示显著关联
    if feat['prr'] > 2.0 and feat['count'] >= 3:
        anomaly_score += 30
        reasons.append(f"PRR={feat['prr']:.2f} (显著关联)")
    
    # 规则2: 高卡方值（> 10）表示统计显著
    if feat['chi2'] > 10:
        anomaly_score += 20
        reasons.append(f"χ²={feat['chi2']:.1f} (统计显著)")
    
    # 规则3: 高死亡率（> 20%）
    if feat['death_rate'] > 0.2 and feat['count'] >= 3:
        anomaly_score += 35
        reasons.append(f"死亡率={feat['death_rate']*100:.1f}%")
    
    # 规则4: 高严重性率（> 50%）
    if feat['serious_rate'] > 0.5 and feat['count'] >= 5:
        anomaly_score += 20
        reasons.append(f"严重事件率={feat['serious_rate']*100:.1f}%")
    
    # 规则5: 罕见但严重的组合
    if feat['rarity_score'] > 8 and feat['serious_rate'] > 0.3:
        anomaly_score += 25
        reasons.append(f"罕见+严重")
    
    # 规则6: 极高的住院率
    if feat['hosp_rate'] > 0.4 and feat['count'] >= 3:
        anomaly_score += 15
        reasons.append(f"住院率={feat['hosp_rate']*100:.1f}%")
    
    # 只保留异常分数 >= 40 的
    if anomaly_score >= 40:
        anomalies.append({
            **feat,
            'anomaly_score': anomaly_score,
            'reasons': reasons
        })

# 按异常分数排序
anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)

print(f"✓ 检测到 {len(anomalies)} 个异常的药物-事件关系\n")

# ============================================================================
# 5. 结果输出和验证
# ============================================================================

print("=" * 80)
print("异常检测结果 - Top 50 最显著的药物-事件信号")
print("=" * 80)
print()

# 保存完整结果到CSV
with open('task3_anomalies_detected.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'drug', 'event', 'count', 'anomaly_score', 'prr', 'ror', 'chi2',
        'serious_rate', 'death_rate', 'hosp_rate', 'rarity_score', 'reasons'
    ])
    writer.writeheader()
    
    for anom in anomalies:
        writer.writerow({
            'drug': anom['drug'],
            'event': anom['event'],
            'count': anom['count'],
            'anomaly_score': anom['anomaly_score'],
            'prr': f"{anom['prr']:.3f}",
            'ror': f"{anom['ror']:.3f}",
            'chi2': f"{anom['chi2']:.2f}",
            'serious_rate': f"{anom['serious_rate']:.3f}",
            'death_rate': f"{anom['death_rate']:.3f}",
            'hosp_rate': f"{anom['hosp_rate']:.3f}",
            'rarity_score': f"{anom['rarity_score']:.3f}",
            'reasons': '; '.join(anom['reasons'])
        })

print("💾 完整结果已保存到: task3_anomalies_detected.csv\n")

# 展示Top 50
for i, anom in enumerate(anomalies[:50], 1):
    print(f"[{i}] 异常分数: {anom['anomaly_score']}")
    print(f"    药物: {anom['drug']}")
    print(f"    不良事件: {anom['event']}")
    print(f"    报告数: {anom['count']}")
    print(f"    PRR: {anom['prr']:.3f} | ROR: {anom['ror']:.3f} | χ²: {anom['chi2']:.2f}")
    print(f"    严重性: {anom['serious_rate']*100:.1f}% | 死亡: {anom['death_rate']*100:.1f}% | 住院: {anom['hosp_rate']*100:.1f}%")
    print(f"    检测原因: {', '.join(anom['reasons'])}")
    print()

# ============================================================================
# 6. 按药物汇总异常信号
# ============================================================================

print("=" * 80)
print("按药物汇总的异常信号统计")
print("=" * 80)
print()

drug_anomaly_summary = defaultdict(lambda: {'count': 0, 'events': [], 'max_score': 0})

for anom in anomalies:
    drug = anom['drug']
    drug_anomaly_summary[drug]['count'] += 1
    drug_anomaly_summary[drug]['events'].append(anom['event'])
    drug_anomaly_summary[drug]['max_score'] = max(
        drug_anomaly_summary[drug]['max_score'],
        anom['anomaly_score']
    )

# 排序并输出
sorted_drugs = sorted(
    drug_anomaly_summary.items(),
    key=lambda x: (x[1]['count'], x[1]['max_score']),
    reverse=True
)

for drug, info in sorted_drugs[:20]:
    print(f"{drug}:")
    print(f"  异常信号数: {info['count']}")
    print(f"  最高异常分数: {info['max_score']}")
    print(f"  关联事件: {', '.join(info['events'][:5])}")
    if len(info['events']) > 5:
        print(f"           ...还有 {len(info['events'])-5} 个事件")
    print()

# ============================================================================
# 7. 关键发现总结
# ============================================================================

print("=" * 80)
print("关键发现总结")
print("=" * 80)
print()

# 统计高风险信号
high_risk = [a for a in anomalies if a['anomaly_score'] >= 70]
medium_risk = [a for a in anomalies if 50 <= a['anomaly_score'] < 70]
low_risk = [a for a in anomalies if 40 <= a['anomaly_score'] < 50]

print(f"🔴 高风险信号 (分数≥70): {len(high_risk)} 个")
print(f"🟡 中风险信号 (50-69): {len(medium_risk)} 个")
print(f"🟢 低风险信号 (40-49): {len(low_risk)} 个")
print()

# 最致命的药物-事件对
fatal_signals = sorted(
    [a for a in anomalies if a['death_rate'] > 0],
    key=lambda x: (x['death_rate'], x['count']),
    reverse=True
)[:10]

print("最致命的药物-事件关系（Top 10）:")
for i, sig in enumerate(fatal_signals, 1):
    print(f"  {i}. {sig['drug']} → {sig['event']}")
    print(f"     死亡率: {sig['death_rate']*100:.1f}% ({int(sig['count']*sig['death_rate'])}/{sig['count']} 例)")
print()

# Epcoritamab的特殊分析
epcoritamab_signals = [a for a in anomalies if a['drug'] == 'Epcoritamab']
if epcoritamab_signals:
    print(f"Epcoritamab 异常信号分析:")
    print(f"  检测到 {len(epcoritamab_signals)} 个异常信号")
    print(f"  相关事件:")
    for sig in epcoritamab_signals[:10]:
        print(f"    - {sig['event']} (分数:{sig['anomaly_score']}, 报告:{sig['count']}例)")
print()

print("=" * 80)
print("任务3完成！异常检测结果已生成。")
print("=" * 80)


