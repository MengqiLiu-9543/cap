#!/usr/bin/env python3
"""
Requirement 2: 真实数据收集器
从FAERS API收集真实的生存分析数据
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

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

def collect_drug_events(drug_name, limit=100):
    """
    为指定药物收集不良事件数据（用于生存分析）
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
                    processed = process_survival_record(record, drug_name)
                    if processed:
                        all_records.extend(processed)
                
                print(f"  进度: {len(all_records)}/{limit}")
                skip += len(results)
                
                # API限流
                time.sleep(0.5)
                
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

def process_survival_record(record, target_drug):
    """
    提取生存分析所需的关键字段
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
        weight = patient.get('patientweight', '')
        
        # 死亡信息
        death_info = patient.get('patientdeath', {})
        death_date = death_info.get('patientdeathdate', '')
        
        # 严重性指标
        serious = record.get('serious', 0)
        seriousness_death = record.get('seriousnessdeath', 0)
        seriousness_hosp = record.get('seriousnesshospitalization', 0)
        seriousness_life = record.get('seriousnesslifethreatening', 0)
        seriousness_disable = record.get('seriousnessdisabling', 0)
        
        # 提取所有药物（包含时间信息）
        drugs = patient.get('drug', [])
        drug_data = []
        target_drug_data = None
        
        for drug in drugs:
            openfda = drug.get('openfda', {})
            generic_names = openfda.get('generic_name', [])
            
            drug_info = {
                'generic_name': '|'.join(generic_names) if generic_names else '',
                'start_date': drug.get('drugstartdate', ''),
                'end_date': drug.get('drugenddate', ''),
                'treatment_duration': drug.get('drugtreatmentduration', ''),
                'treatment_duration_unit': drug.get('drugtreatmentdurationunit', ''),
                'indication': drug.get('drugindication', ''),
                'dosage_form': drug.get('drugdosageform', ''),
                'administration_route': drug.get('drugadministrationroute', ''),
                'characterization': drug.get('drugcharacterization', ''),
            }
            
            drug_data.append(drug_info)
            
            # 检查是否为目标药物
            if target_drug.lower() in [name.lower() for name in generic_names]:
                target_drug_data = drug_info
        
        # 提取所有不良事件
        reactions = patient.get('reaction', [])
        adverse_events = []
        reaction_outcomes = []
        
        for reaction in reactions:
            ae_term = reaction.get('reactionmeddrapt', '')
            outcome = reaction.get('reactionoutcome', '')
            if ae_term:
                adverse_events.append(ae_term)
                reaction_outcomes.append(outcome)
        
        # 计算时间到事件
        time_to_event = calculate_time_to_event(target_drug_data, receive_date, death_date, serious)
        
        # 分类不良事件类型
        processed_records = []
        
        for i, ae in enumerate(adverse_events):
            event_type = classify_adverse_event(ae)
            
            # 提取风险因素
            age_group = classify_age_group(age, age_unit)
            weight_group = classify_weight_group(weight)
            polypharmacy = len(drugs) > 3
            
            record_data = {
                'safety_report_id': safety_id,
                'receive_date': receive_date,
                'target_drug': target_drug,
                'adverse_event': ae,
                'event_type': event_type,
                'is_long_term_event': 1 if event_type in ['INFECTION', 'SECONDARY_MALIGNANCY'] else 0,
                'is_infection': 1 if event_type == 'INFECTION' else 0,
                'is_secondary_malignancy': 1 if event_type == 'SECONDARY_MALIGNANCY' else 0,
                
                # 时间到事件数据
                'time_to_event_days': time_to_event,
                'event_occurred': 1 if time_to_event > 0 else 0,
                'censored': 1 if serious == 0 else 0,
                
                # 患者人口统计学
                'patient_age': age,
                'patient_age_unit': age_unit,
                'patient_sex': sex,
                'patient_weight': weight,
                
                # 严重性指标
                'is_serious': serious,
                'is_death': seriousness_death,
                'is_hospitalization': seriousness_hosp,
                'is_lifethreatening': seriousness_life,
                'is_disabling': seriousness_disable,
                
                # 药物治疗信息
                'drug_start_date': target_drug_data.get('start_date', '') if target_drug_data else '',
                'drug_end_date': target_drug_data.get('end_date', '') if target_drug_data else '',
                'treatment_duration': target_drug_data.get('treatment_duration', '') if target_drug_data else '',
                'treatment_duration_unit': target_drug_data.get('treatment_duration_unit', '') if target_drug_data else '',
                'drug_indication': target_drug_data.get('indication', '') if target_drug_data else '',
                'administration_route': target_drug_data.get('administration_route', '') if target_drug_data else '',
                'drug_characterization': target_drug_data.get('characterization', '') if target_drug_data else '',
                
                # 风险因素
                'age_group': age_group,
                'weight_group': weight_group,
                'polypharmacy': polypharmacy,
                'multiple_events': len(adverse_events) > 1,
                'concomitant_drugs': max(0, len(drugs) - 1),
                'drug_interaction_risk': 'LOW' if len(drugs) <= 2 else 'MEDIUM' if len(drugs) <= 5 else 'HIGH',
                
                # 额外特征
                'total_drugs': len(drugs),
                'total_events': len(adverse_events),
                'reaction_outcome': reaction_outcomes[i] if i < len(reaction_outcomes) else '',
                'death_date': death_date
            }
            
            processed_records.append(record_data)
        
        return processed_records
        
    except Exception as e:
        print(f"    记录处理错误: {str(e)}")
        return []

def calculate_time_to_event(drug_data, receive_date, death_date, serious):
    """
    计算时间到事件（天数）
    """
    try:
        if not drug_data:
            return 0.0
        
        drug_start = drug_data.get('start_date', '')
        drug_end = drug_data.get('end_date', '')
        
        # 解析日期
        start_date = parse_date(drug_start) if drug_start else None
        end_date = parse_date(drug_end) if drug_end else None
        report_date = parse_date(receive_date) if receive_date else None
        death_dt = parse_date(death_date) if death_date else None
        
        # 计算时间到事件
        if start_date and report_date:
            if death_dt and serious:
                time_diff = (death_dt - start_date).days
            elif end_date:
                time_diff = (end_date - start_date).days
            else:
                time_diff = (report_date - start_date).days
            
            return max(0, min(3650, time_diff))  # 限制在0-3650天（10年）
        
        return 0.0
        
    except Exception:
        return 0.0

def parse_date(date_str):
    """
    解析日期字符串
    """
    if not date_str or str(date_str).strip() == '':
        return None
    
    try:
        date_str = str(date_str).strip()
        
        # 尝试不同的日期格式
        formats = ['%Y%m%d', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
        
    except Exception:
        return None

def classify_adverse_event(ae_term):
    """
    分类不良事件
    """
    ae_upper = ae_term.upper()
    
    infection_terms = ['INFECTION', 'SEPSIS', 'PNEUMONIA', 'BACTERIAL', 'FUNGAL', 
                       'VIRAL', 'OPPORTUNISTIC', 'NEUTROPENIC', 'FEBRILE']
    
    for term in infection_terms:
        if term in ae_upper:
            return 'INFECTION'
    
    malignancy_terms = ['MALIGNANCY', 'NEOPLASM', 'CANCER', 'CARCINOMA', 
                        'SARCOMA', 'LYMPHOMA', 'LEUKEMIA', 'MYELOMA']
    
    for term in malignancy_terms:
        if term in ae_upper:
            return 'SECONDARY_MALIGNANCY'
    
    return 'OTHER'

def classify_age_group(age, age_unit):
    """
    分类年龄组
    """
    if not age or age == '':
        return 'UNKNOWN'
    
    try:
        age_val = float(age)
        age_unit_upper = str(age_unit).upper() if age_unit else ''
        
        if 'YEAR' in age_unit_upper or age_unit_upper == '':
            if age_val < 18:
                return 'PEDIATRIC'
            elif age_val < 65:
                return 'ADULT'
            else:
                return 'ELDERLY'
        elif 'MONTH' in age_unit_upper:
            if age_val < 12:
                return 'INFANT'
            else:
                return 'PEDIATRIC'
    except:
        pass
    
    return 'UNKNOWN'

def classify_weight_group(weight):
    """
    分类体重组
    """
    if not weight or weight == '':
        return 'UNKNOWN'
    
    try:
        weight_val = float(weight)
        if weight_val < 50:
            return 'UNDERWEIGHT'
        elif weight_val < 100:
            return 'NORMAL'
        else:
            return 'OVERWEIGHT'
    except:
        pass
    
    return 'UNKNOWN'

def main():
    """
    主函数：收集所有肿瘤药物的生存分析数据
    """
    print("=" * 80)
    print("Requirement 2: 真实数据收集 - 从FAERS API")
    print("=" * 80)
    print(f"目标药物数量: {len(ONCOLOGY_DRUGS)}")
    print(f"每种药物收集记录数: 100\n")
    
    all_data = []
    
    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f"[{i}/{len(ONCOLOGY_DRUGS)}] {drug}")
        records = collect_drug_events(drug, limit=100)
        
        # 展开嵌套列表
        for record_list in records:
            if isinstance(record_list, list):
                all_data.extend(record_list)
            else:
                all_data.append(record_list)
        
        # 每10个药物保存一次中间结果
        if i % 10 == 0:
            temp_df = pd.DataFrame(all_data)
            temp_df.to_csv(f'requirement2_data_temp_{i}.csv', index=False)
            print(f"💾 中间结果已保存: {len(all_data)} 条记录\n")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 数据清洗
    df = df.drop_duplicates()
    df = df.dropna(subset=['adverse_event'])  # 必须有不良事件
    
    # 保存结果
    output_file = 'requirement2_faers_data.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("数据收集完成！")
    print("=" * 80)
    print(f"总记录数: {len(df)}")
    print(f"唯一药物数: {df['target_drug'].nunique()}")
    print(f"唯一不良事件数: {df['adverse_event'].nunique()}")
    print(f"严重事件比例: {df['is_serious'].sum() / len(df) * 100:.1f}%")
    print(f"死亡事件数: {df['is_death'].sum()}")
    print(f"长期事件数: {df['is_long_term_event'].sum()}")
    print(f"感染事件数: {df['is_infection'].sum()}")
    print(f"继发恶性肿瘤数: {df['is_secondary_malignancy'].sum()}")
    print(f"输出文件: {output_file}")
    
    # 显示前几行示例
    print("\n数据样本（前5行）:")
    print(df[['target_drug', 'adverse_event', 'is_serious', 'time_to_event_days', 'patient_age']].head())
    
    # 统计信息
    print("\n不良事件频率（Top 20）:")
    print(df['adverse_event'].value_counts().head(20))
    
    print("\n药物事件数量统计:")
    print(df['target_drug'].value_counts())

if __name__ == "__main__":
    main()

