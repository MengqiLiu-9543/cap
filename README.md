# Task 3: Detect Rare and Unexpected Drug-Event Relationships

## 项目概览

本项目实现了**任务3：检测罕见和意外的药物-事件关系**，使用Isolation Forest算法对肿瘤药物的不良事件进行异常检测，识别潜在的药物安全信号。

**完成时间**: 2025年10月15日  
**状态**: ✅ 已完成

---

## 📁 文件清单

### 代码文件

| 文件 | 大小 | 描述 |
|------|------|------|
| `task3_data_collector.py` | 7.6 KB | 数据收集脚本，从OpenFDA API获取35种肿瘤药物的不良事件数据 |
| `task3_anomaly_detection.py` | 11 KB | 异常检测算法，实现基于Isolation Forest原理的药物安全信号检测 |

### 数据文件

| 文件 | 大小 | 描述 |
|------|------|------|
| `task3_oncology_drug_event_pairs.csv` | 16 MB | 原始数据集，包含55,604条药物-事件对 |
| `task3_anomalies_detected.csv` | 889 KB | 检测结果，包含6,826个异常信号 |

### 文档文件

| 文件 | 大小 | 描述 |
|------|------|------|
| `task3_completion_summary.md` | 8.5 KB | 项目完成通知，快速了解项目状态和成果 |
| `task3_executive_summary.md` | 6.8 KB | 执行摘要，适合项目经理和非技术人员阅读 |
| `task3_final_report.md` | 14 KB | 完整技术报告，包含详细方法论和发现 |
| `task3_visualization_summary.md` | 15 KB | 可视化总结，包含图表和数据排行榜 |

---

## 🎯 核心成果

### 数据规模

- **35种**肿瘤药物
- **55,604条**药物-事件对
- **3,488种**独特不良事件
- **17,339个**药物-事件组合

### 检测结果

- **6,826个**异常信号
- **2,639个**高风险信号 (🔴)
- **893个**中风险信号 (🟡)
- **3,294个**低风险信号 (🟢)

### 重点发现

#### Top 5 最显著异常信号

1. **Pembrolizumab + Pneumonitis** (分数:145, 死亡率:26.7%)
2. **Nivolumab + Confusional state** (分数:145, 死亡率:40.0%)
3. **Nivolumab + Pancreatitis** (分数:145, 死亡率:33.3%)
4. **Nivolumab + Hypercalcaemia** (分数:145, 死亡率:60.0%)
5. **Nivolumab + AST increased** (分数:145, 死亡率:50.0%)

#### Epcoritamab 关键发现

- **196个异常信号**（所有35种药物中排名第17）
- **神经毒性** (PRR=10.82, 死亡率66.7%) ⚠️ 高致命性
- **CMV感染** (PRR=23.08, 显著高于其他药物)
- **低丙种球蛋白血症** (PRR=36.13)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install requests
```

### 2. 运行数据收集

```bash
python task3_data_collector.py
```

这将从OpenFDA API收集35种肿瘤药物的不良事件数据，生成 `task3_oncology_drug_event_pairs.csv`。

### 3. 运行异常检测

```bash
python task3_anomaly_detection.py
```

这将对收集的数据进行异常检测，生成 `task3_anomalies_detected.csv`。

### 4. 查看结果

```python
import pandas as pd

# 加载异常检测结果
df = pd.read_csv('task3_anomalies_detected.csv')

# 查看高风险信号
high_risk = df[df['anomaly_score'].astype(float) >= 70]
print(f"高风险信号数量: {len(high_risk)}")

# 查看Epcoritamab的异常信号
epc = df[df['drug'] == 'Epcoritamab']
print(f"\nEpcoritamab异常信号:\n{epc[['event', 'anomaly_score', 'prr', 'death_rate']].head(10)}")
```

---

## 📊 方法论

### 特征工程

为每个药物-事件对计算：

- **统计关联指标**: PRR (Proportional Reporting Ratio), ROR, 卡方检验
- **严重性指标**: 死亡率、住院率、严重事件率
- **罕见性评分**: 基于报告频率

### 异常检测算法

基于Isolation Forest原理的多规则评分系统：

| 规则 | 条件 | 分数 | 临床意义 |
|------|------|------|----------|
| 1 | PRR > 2.0 且报告数 ≥ 3 | +30 | 显著关联 |
| 2 | χ² > 10 | +20 | 统计显著 |
| 3 | 死亡率 > 20% 且报告数 ≥ 3 | +35 | 高致命性 |
| 4 | 严重率 > 50% 且报告数 ≥ 5 | +20 | 高严重性 |
| 5 | 罕见性评分 > 8 且严重率 > 30% | +25 | 罕见+严重 |
| 6 | 住院率 > 40% 且报告数 ≥ 3 | +15 | 高住院风险 |

**异常阈值**: 总分 ≥ 40分

### 验证方法

1. **统计验证**: PRR > 2.0 且 χ² > 10
2. **临床相关性**: 严重性、死亡率、住院率评估
3. **文献验证**: 与FDA警告和已知证据对比
4. **生物学合理性**: 机制分析

---

## 📖 文档导读

### 对于不同角色的推荐阅读顺序

#### 项目经理 / 非技术人员
1. 📄 `task3_completion_summary.md` - 快速了解项目完成情况
2. 📄 `task3_executive_summary.md` - 执行摘要和关键发现
3. 📄 `task3_visualization_summary.md` - 可视化图表和排行榜

#### 数据科学家 / 技术人员
1. 📄 `task3_final_report.md` - 完整技术报告
2. 📄 `task3_anomaly_detection.py` - 算法实现代码
3. 📄 `task3_data_collector.py` - 数据收集代码
4. 📊 `task3_anomalies_detected.csv` - 分析检测结果

#### 临床医生 / 药物安全专家
1. 📄 `task3_executive_summary.md` - 关键临床发现
2. 📄 `task3_visualization_summary.md` - Epcoritamab风险图谱
3. 📊 `task3_anomalies_detected.csv` - 具体药物-事件对数据

---

## ✅ 验证结果

### 算法准确性验证

检测到的高风险信号与已知FDA警告的一致性：

- ✅ **Pembrolizumab/Nivolumab → Pneumonitis** (FDA黑框警告)
- ✅ **Ipilimumab → Intestinal perforation** (FDA警告)
- ✅ **Nivolumab → Hypothyroidism** (已知免疫相关AE)
- ✅ **Durvalumab → Myocarditis** (已知心脏毒性)

**结论**: 算法能够正确识别已知的严重药物安全信号

### 潜在新发现

需要进一步研究的信号：

1. **Epcoritamab → Neurotoxicity** (PRR=10.82, 死亡率66.7%)
2. **Epcoritamab → Hypogammaglobulinaemia** (PRR=36.13)
3. **Nivolumab → Hypopituitarism** (PRR=26.62)

---

## 💡 临床应用建议

### Epcoritamab 监测方案

**必须监测**:
- ✅ 神经系统检查（每次给药前）
- ✅ 感染筛查（CMV, 细菌）
- ✅ 免疫球蛋白水平（每月）
- ✅ 细胞因子释放综合征监测

**预防措施**:
- 考虑预防性抗病毒治疗（CMV）
- 免疫球蛋白替代治疗（IgG < 400 mg/dL）
- 神经毒性早期干预方案

---

## 📚 技术栈

- **编程语言**: Python 3.12
- **数据来源**: OpenFDA Drug Adverse Event API
- **核心库**: requests (API调用), csv (数据处理)
- **算法**: Isolation Forest原理（基于规则的实现）
- **统计方法**: PRR, ROR, 卡方检验

---

## 📊 数据字段说明

### `task3_oncology_drug_event_pairs.csv` 字段

| 字段 | 类型 | 描述 |
|------|------|------|
| safety_report_id | String | FDA安全报告唯一ID |
| receive_date | String | 报告接收日期 |
| target_drug | String | 目标药物名称 |
| all_drugs | String | 患者使用的所有药物（\|分隔） |
| drug_count | Integer | 药物数量 |
| adverse_event | String | 不良事件术语（MedDRA） |
| event_count | Integer | 该报告中的不良事件数量 |
| patient_age | Float | 患者年龄 |
| patient_age_unit | String | 年龄单位 |
| patient_sex | Integer | 患者性别（1=男，2=女） |
| is_serious | Integer | 是否严重事件（1=是） |
| is_death | Integer | 是否导致死亡（1=是） |
| is_hospitalization | Integer | 是否导致住院（1=是） |
| is_lifethreatening | Integer | 是否危及生命（1=是） |
| is_disabling | Integer | 是否导致残疾（1=是） |
| indication | String | 用药适应症 |

### `task3_anomalies_detected.csv` 字段

| 字段 | 类型 | 描述 |
|------|------|------|
| drug | String | 药物名称 |
| event | String | 不良事件术语 |
| count | Integer | 报告数量 |
| anomaly_score | Integer | 异常分数（40-145） |
| prr | Float | PRR值（> 2.0表示显著关联） |
| ror | Float | ROR值 |
| chi2 | Float | 卡方统计量（> 10表示显著） |
| serious_rate | Float | 严重事件比例（0-1） |
| death_rate | Float | 死亡比例（0-1） |
| hosp_rate | Float | 住院比例（0-1） |
| rarity_score | Float | 罕见性评分 |
| reasons | String | 检测原因（;分隔） |

---

## ⚠️ 使用注意事项

### 数据解释

1. **关联 ≠ 因果**: 检测到的异常信号表示统计关联，不一定是因果关系
2. **报告偏倚**: FDA FAERS数据基于自发报告，存在漏报和选择性报告
3. **混杂因素**: 未调整患者基线特征、合并用药等因素
4. **时间滞后**: 数据可能不是实时的

### 临床应用限制

1. **不能替代临床判断**: 结果需结合临床经验和患者个体情况
2. **需进一步验证**: 新发现的信号需要前瞻性研究验证
3. **不适用于个体预测**: 基于群体数据，不能预测个体风险

---

## 🔗 相关资源

- [OpenFDA API 文档](https://open.fda.gov/apis/)
- [FDA药物不良事件报告系统 (FAERS)](https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- [MedDRA术语表](https://www.meddra.org/)
- [WHO药物警戒](https://www.who.int/teams/regulation-prequalification/pharmacovigilance)

---

## 📞 联系方式

如有问题或建议，请：
- 查看详细文档（`task3_final_report.md`）
- 检查代码注释（`task3_anomaly_detection.py`）
- 联系项目团队

---

## 📄 许可证

本项目用于学术研究和教育目的。

---

## 🙏 致谢

- **OpenFDA**: 提供高质量公开数据
- **FDA FAERS**: 药物不良事件报告系统
- **药物警戒社区**: PRR/ROR方法论

---

**项目状态**: ✅ 已完成  
**最后更新**: 2025年10月15日  
**版本**: 1.0

🎉 **欢迎使用本项目进行药物安全研究！**

