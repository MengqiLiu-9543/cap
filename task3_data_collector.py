#!/usr/bin/env python3
"""
任务3数据收集器：检测罕见和意外的药物-事件关系
收集结构化的药物-事件关联数据，用于异常检测

目标：
- 收集肿瘤药物的不良事件数据
- 提取关键字段：药物名称、不良事件、严重性、患者信息、时间
- 生成适合异常检测的结构化数据集
"""

import requests
import pandas as pd
import json
import time
from collections import defaultdict

# 35种常见肿瘤药物（聚焦靶向治疗和免疫治疗）
ONCOLOGY_DRUGS = [
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab", "Ipilimumab",
    "Trastuzumab", "Bevacizumab", "Cetuximab", "Rituximab", "Epcoritamab",
    "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib", "Crizotinib",
    "Paclitaxel", "Docetaxel", "Doxorubicin", "Carboplatin", "Cisplatin",
    "Lenalidomide", "Pomalidomide", "Bortezomib", "Carfilzomib", "Venetoclax",
    "Ibrutinib", "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
    "Palbociclib", "Ribociclib", "Abemaciclib", "Vemurafenib", "Dabrafenib"
]

BASE_URL = "https://api.fda.gov/drug/event.json"

def collect_drug_events(drug_name, limit=500):
    """
    为指定药物收集不良事件数据
    """
    print(f"正在收集 {drug_name} 的数据...")
    
    all_records = []
    skip = 0
    
    while len(all_records) < limit:
        try:
            # 构建查询
            params = {
                'search': f'patient.drug.openfda.generic_name:"{drug_name}"',
                'limit': min(100, limit - len(all_records)),
                'skip': skip
            }
            
            response = requests.get(BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    print(f"  ✓ {drug_name}: 未找到更多数据，共收集 {len(all_records)} 条")
                    break
                
                # 处理每条记录
                for record in results:
                    processed = process_record(record, drug_name)
                    if processed:
                        all_records.append(processed)
                
                print(f"  进度: {len(all_records)}/{limit}")
                skip += len(results)
                
                # API限流
                time.sleep(0.3)
                
            elif response.status_code == 404:
                print(f"  ⚠ {drug_name}: 未找到数据")
                break
            else:
                print(f"  ✗ {drug_name}: HTTP {response.status_code}")
                break
                
        except Exception as e:
            print(f"  ✗ {drug_name}: 错误 - {str(e)}")
            break
    
    print(f"  ✓ {drug_name}: 完成，共 {len(all_records)} 条记录\n")
    return all_records

def process_record(record, target_drug):
    """
    提取关键字段，生成结构化记录
    """
    try:
        # 基本信息
        safety_id = record.get('safetyreportid', '')
        receive_date = record.get('receivedate', '')
        
        # 患者信息
        patient = record.get('patient', {})
        age = patient.get('patientonsetage', '')
        age_unit = patient.get('patientonsetageunit', '')
        sex = patient.get('patientsex', '')
        
        # 严重性指标
        serious = record.get('serious', 0)
        seriousness_death = record.get('seriousnessdeath', 0)
        seriousness_hosp = record.get('seriousnesshospitalization', 0)
        seriousness_life = record.get('seriousnesslifethreatening', 0)
        seriousness_disable = record.get('seriousnessdisabling', 0)
        
        # 提取所有药物
        drugs = patient.get('drug', [])
        drug_names = []
        drug_indications = []
        for drug in drugs:
            openfda = drug.get('openfda', {})
            generic_names = openfda.get('generic_name', [])
            drug_names.extend(generic_names)
            indication = drug.get('drugindication', '')
            if indication:
                drug_indications.append(indication)
        
        # 提取所有不良事件（MedDRA术语）
        reactions = patient.get('reaction', [])
        adverse_events = []
        for reaction in reactions:
            ae_term = reaction.get('reactionmeddrapt', '')
            if ae_term:
                adverse_events.append(ae_term)
        
        # 为每个不良事件创建一条记录（药物-事件对）
        processed_records = []
        for ae in adverse_events:
            processed_records.append({
                'safety_report_id': safety_id,
                'receive_date': receive_date,
                'target_drug': target_drug,
                'all_drugs': '|'.join(drug_names),
                'drug_count': len(drugs),
                'adverse_event': ae,
                'event_count': len(adverse_events),
                'patient_age': age,
                'patient_age_unit': age_unit,
                'patient_sex': sex,
                'is_serious': serious,
                'is_death': seriousness_death,
                'is_hospitalization': seriousness_hosp,
                'is_lifethreatening': seriousness_life,
                'is_disabling': seriousness_disable,
                'indication': '|'.join(drug_indications)
            })
        
        return processed_records
        
    except Exception as e:
        print(f"    记录处理错误: {str(e)}")
        return None

def main():
    """
    主函数：收集所有肿瘤药物的数据
    """
    print("=" * 80)
    print("任务3：罕见和意外药物-事件关系检测 - 数据收集器")
    print("=" * 80)
    print(f"目标药物数量: {len(ONCOLOGY_DRUGS)}")
    print(f"每种药物收集记录数: 500\n")
    
    all_data = []
    
    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f"[{i}/{len(ONCOLOGY_DRUGS)}] {drug}")
        records = collect_drug_events(drug, limit=500)
        
        # 展开嵌套列表
        for record_list in records:
            if isinstance(record_list, list):
                all_data.extend(record_list)
            else:
                all_data.append(record_list)
        
        # 每10个药物保存一次中间结果
        if i % 10 == 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_csv(f'task3_data_temp_{i}.csv', index=False)
            print(f"💾 中间结果已保存: {len(all_data)} 条记录\n")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 数据清洗
    df = df.drop_duplicates()
    df = df.dropna(subset=['adverse_event'])  # 必须有不良事件
    
    # 保存结果
    output_file = 'task3_oncology_drug_event_pairs.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("数据收集完成！")
    print("=" * 80)
    print(f"总记录数: {len(df)}")
    print(f"唯一药物数: {df['target_drug'].nunique()}")
    print(f"唯一不良事件数: {df['adverse_event'].nunique()}")
    print(f"严重事件比例: {df['is_serious'].sum() / len(df) * 100:.1f}%")
    print(f"死亡事件数: {df['is_death'].sum()}")
    print(f"输出文件: {output_file}")
    
    # 显示前几行示例
    print("\n数据样本（前5行）:")
    print(df[['target_drug', 'adverse_event', 'is_serious', 'is_death', 'patient_age', 'patient_sex']].head())
    
    # 统计信息
    print("\n不良事件频率（Top 20）:")
    print(df['adverse_event'].value_counts().head(20))
    
    print("\n药物事件数量统计:")
    print(df['target_drug'].value_counts())

if __name__ == "__main__":
    main()


