#!/usr/bin/env python3
"""
Task 5 数据收集器：预测不良事件严重程度

参考Task 3的提取策略，针对Task 5优化：
- 收集35种常见肿瘤药物的不良事件数据
- 每种药物提取500-1000条记录
- 保持完整的报告级别数据（不展开为事件对）
- 预计总数据量: 15,000-20,000条

Task 5特点：
- 目标是预测严重程度（死亡、住院等）
- 需要完整的患者和药物信息
- 需要保持报告的完整性（一个报告一条记录）
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

print("=" * 80)
print("Task 5: 不良事件严重程度预测 - 数据收集器")
print("=" * 80)
print()

# 35种常见肿瘤药物（参考Task 3，涵盖主要治疗类别）
ONCOLOGY_DRUGS = [
    # PD-1/PD-L1免疫检查点抑制剂
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab", "Ipilimumab",
    
    # 单克隆抗体靶向治疗
    "Trastuzumab", "Bevacizumab", "Cetuximab", "Rituximab", "Epcoritamab",
    "Pertuzumab", "Panitumumab", "Ramucirumab", "Daratumumab",
    
    # 小分子靶向药物（TKI）
    "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib", "Crizotinib",
    "Palbociclib", "Ribociclib", "Abemaciclib", "Vemurafenib", "Dabrafenib",
    "Ibrutinib", "Venetoclax",
    
    # PARP抑制剂
    "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
    
    # 化疗药物（常用）
    "Paclitaxel", "Docetaxel", "Doxorubicin",
    
    # 免疫调节剂
    "Lenalidomide", "Pomalidomide"
]

BASE_URL = "https://api.fda.gov/drug/event.json"
MAX_RETRIES = 3
RETRY_DELAY = 5

def collect_drug_data(drug_name, max_records=500):
    """
    为指定药物收集不良事件数据（完整报告）
    """
    print(f"\n📥 提取药物: {drug_name}")
    
    all_records = []
    skip = 0
    limit = 100
    retries = 0
    
    while len(all_records) < max_records:
        try:
            # 构建查询
            params = {
                'search': f'patient.drug.openfda.generic_name:"{drug_name}"',
                'limit': min(limit, max_records - len(all_records)),
                'skip': skip
            }
            
            response = requests.get(BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    print(f"   未找到更多数据，共收集 {len(all_records)} 条")
                    break
                
                # 处理每条记录
                for record in results:
                    processed = process_record(record, drug_name)
                    if processed:
                        all_records.append(processed)
                
                print(f"   进度: {len(all_records)}", end='\r')
                skip += len(results)
                retries = 0  # 重置重试计数
                
                # API限流
                time.sleep(0.3)
                
            elif response.status_code == 404:
                print(f"   ⚠️  未找到数据")
                break
            else:
                retries += 1
                if retries >= MAX_RETRIES:
                    print(f"   ❌ HTTP {response.status_code}，达到最大重试次数")
                    break
                print(f"   ⚠️  HTTP {response.status_code}，重试 {retries}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
                
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                print(f"   ❌ 错误: {str(e)[:50]}，达到最大重试次数")
                break
            print(f"   ⚠️  错误: {str(e)[:50]}，重试 {retries}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)
    
    print(f"   ✅ 完成: {len(all_records)} 条记录")
    return all_records

def process_record(record, target_drug):
    """
    提取关键字段，保持报告级别的完整性
    """
    try:
        # 基本信息
        safety_id = record.get('safetyreportid', '')
        receive_date = record.get('receivedate', '')
        
        # 患者信息
        patient = record.get('patient', {})
        age = patient.get('patientonsetage', None)
        age_unit = patient.get('patientonsetageunit', '')
        sex = patient.get('patientsex', 0)
        weight = patient.get('patientweight', None)
        
        # 严重性指标（Task 5的目标变量）
        serious = record.get('serious', 0)
        seriousness_death = record.get('seriousnessdeath', 0)
        seriousness_hosp = record.get('seriousnesshospitalization', 0)
        seriousness_life = record.get('seriousnesslifethreatening', 0)
        seriousness_disable = record.get('seriousnessdisabling', 0)
        seriousness_congenital = record.get('seriousnesscongenitalanomali', 0)
        seriousness_other = record.get('seriousnessother', 0)
        
        # 报告来源
        primary_source = record.get('primarysource', {})
        qualification = primary_source.get('qualification', '')
        
        # 药物信息
        drugs = patient.get('drug', [])
        drug_names = []
        drug_roles = []
        drug_indications = []
        
        for drug in drugs:
            openfda = drug.get('openfda', {})
            generic_names = openfda.get('generic_name', [])
            drug_names.extend(generic_names)
            
            # 药物角色（怀疑/伴随）
            role = drug.get('drugcharacterization', '')
            drug_roles.append(role)
            
            indication = drug.get('drugindication', '')
            if indication:
                drug_indications.append(indication)
        
        # 不良事件信息
        reactions = patient.get('reaction', [])
        adverse_events = []
        for reaction in reactions:
            ae_term = reaction.get('reactionmeddrapt', '')
            if ae_term:
                adverse_events.append(ae_term)
        
        # 构建完整的报告记录
        processed_record = {
            # 基本信息
            'safetyreportid': safety_id,
            'receivedate': receive_date,
            'target_drug': target_drug,
            
            # 患者信息
            'patientonsetage': age,
            'patientonsetageunit': age_unit,
            'patientsex': sex,
            'patientweight': weight,
            
            # 严重性指标（目标变量）
            'serious': serious,
            'seriousnessdeath': seriousness_death,
            'seriousnesshospitalization': seriousness_hosp,
            'seriousnesslifethreatening': seriousness_life,
            'seriousnessdisabling': seriousness_disable,
            'seriousnesscongenitalanomali': seriousness_congenital,
            'seriousnessother': seriousness_other,
            
            # 药物信息
            'drugname': target_drug,
            'all_drugs': '|'.join(drug_names) if drug_names else '',
            'num_drugs': len(drugs),
            'drug_indication': '|'.join(drug_indications) if drug_indications else '',
            
            # 不良事件信息
            'reactions': '|'.join(adverse_events) if adverse_events else '',
            'num_reactions': len(adverse_events),
            
            # 报告质量
            'reporter_qualification': qualification
        }
        
        return processed_record
        
    except Exception as e:
        return None

def main():
    """
    主函数：收集所有肿瘤药物的数据
    """
    print(f"📋 数据收集配置:")
    print(f"  目标药物数量: {len(ONCOLOGY_DRUGS)}")
    print(f"  每种药物收集: 500条")
    print(f"  预计总数据量: ~{len(ONCOLOGY_DRUGS) * 500:,}条")
    print(f"  预计耗时: {len(ONCOLOGY_DRUGS) * 2}分钟左右")
    print()
    
    # 显示药物列表
    print("📦 药物列表:")
    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f"  {i:2d}. {drug}")
    print()
    
    response = input("是否开始数据收集？(y/n，默认y): ").strip().lower()
    if response in ['n', 'no']:
        print("已取消")
        return
    
    print()
    print("=" * 80)
    print("🚀 开始数据收集")
    print("=" * 80)
    
    start_time = time.time()
    all_data = []
    success_count = 0
    failed_drugs = []
    
    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f"\n[{i}/{len(ONCOLOGY_DRUGS)}] {drug}")
        print("-" * 80)
        
        try:
            records = collect_drug_data(drug, max_records=500)
            
            if records:
                all_data.extend(records)
                success_count += 1
            else:
                failed_drugs.append(drug)
            
            # 显示累计进度
            print(f"   累计: {len(all_data)} 条 | 成功: {success_count} 药物")
            
            # 每10个药物保存一次中间结果
            if i % 10 == 0:
                temp_df = pd.DataFrame(all_data)
                temp_file = f'task5_data_temp_{i}.csv'
                temp_df.to_csv(temp_file, index=False)
                print(f"   💾 中间结果已保存: {temp_file}")
        
        except Exception as e:
            print(f"   ❌ 处理失败: {str(e)[:50]}")
            failed_drugs.append(drug)
    
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print("📊 数据收集完成")
    print("=" * 80)
    print()
    
    if not all_data:
        print("❌ 错误: 没有收集到任何数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 数据清洗
    print("🔄 数据清洗...")
    print(f"  原始记录: {len(df)}")
    
    # 去重（基于报告ID）
    df = df.drop_duplicates(subset=['safetyreportid'], keep='first')
    print(f"  去重后: {len(df)}")
    
    # 保留有目标变量的记录
    df = df[df['serious'].notna()]
    print(f"  有效记录: {len(df)}")
    print()
    
    # 保存结果
    output_file = 'task5_severity_prediction_data.csv'
    df.to_csv(output_file, index=False)
    
    print("=" * 80)
    print("✅ 数据收集成功！")
    print("=" * 80)
    print()
    
    # 统计信息
    print("📈 数据统计:")
    print(f"  总记录数: {len(df):,}")
    print(f"  唯一报告数: {df['safetyreportid'].nunique():,}")
    print(f"  涉及药物数: {df['target_drug'].nunique()}")
    print(f"  收集成功: {success_count}/{len(ONCOLOGY_DRUGS)} 药物")
    print()
    
    print("🏥 严重程度分布:")
    # 转换为数值类型（OpenFDA可能返回字符串）
    death_count = pd.to_numeric(df['seriousnessdeath'], errors='coerce').fillna(0).sum()
    hosp_count = pd.to_numeric(df['seriousnesshospitalization'], errors='coerce').fillna(0).sum()
    life_count = pd.to_numeric(df['seriousnesslifethreatening'], errors='coerce').fillna(0).sum()
    disable_count = pd.to_numeric(df['seriousnessdisabling'], errors='coerce').fillna(0).sum()
    
    print(f"  死亡案例: {int(death_count):,} ({death_count/len(df)*100:.1f}%)")
    print(f"  住院案例: {int(hosp_count):,} ({hosp_count/len(df)*100:.1f}%)")
    print(f"  危及生命: {int(life_count):,} ({life_count/len(df)*100:.1f}%)")
    print(f"  致残案例: {int(disable_count):,} ({disable_count/len(df)*100:.1f}%)")
    print()
    
    print("👥 患者统计:")
    if 'patientsex' in df.columns:
        sex_counts = df['patientsex'].value_counts()
        sex_map = {1: '男性', 2: '女性', 0: '未知'}
        for sex_code, count in sex_counts.items():
            label = sex_map.get(sex_code, f'代码{sex_code}')
            print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    if 'patientonsetage' in df.columns:
        age_data = pd.to_numeric(df['patientonsetage'], errors='coerce').dropna()
        if len(age_data) > 0:
            print(f"  平均年龄: {age_data.mean():.1f} 岁")
            print(f"  年龄范围: {age_data.min():.1f} - {age_data.max():.1f} 岁")
    print()
    
    print("💊 各药物数据量:")
    drug_counts = df['target_drug'].value_counts()
    for drug, count in drug_counts.head(10).items():
        print(f"  {drug:20s}: {count:5d} 条")
    if len(drug_counts) > 10:
        print(f"  ... (还有 {len(drug_counts)-10} 种药物)")
    print()
    
    if failed_drugs:
        print(f"⚠️  失败的药物 ({len(failed_drugs)}):")
        for drug in failed_drugs:
            print(f"  - {drug}")
        print()
    
    print(f"⏱️  总耗时: {elapsed_time/60:.1f} 分钟")
    print(f"📁 输出文件: {output_file}")
    print()
    
    # 保存统计报告
    with open("task5_collection_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Task 5 数据收集总结\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"收集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed_time/60:.1f} 分钟\n\n")
        
        f.write(f"总记录数: {len(df):,}\n")
        f.write(f"涉及药物: {df['target_drug'].nunique()}\n")
        f.write(f"成功率: {success_count}/{len(ONCOLOGY_DRUGS)} ({success_count/len(ONCOLOGY_DRUGS)*100:.1f}%)\n\n")
        
        f.write("各药物数据量:\n")
        for drug, count in drug_counts.items():
            f.write(f"  {drug:20s}: {count:5d}\n")
        
        if failed_drugs:
            f.write(f"\n失败的药物:\n")
            for drug in failed_drugs:
                f.write(f"  - {drug}\n")
    
    print("💾 已保存统计报告: task5_collection_summary.txt")
    print()
    
    # 显示数据样本
    print("=" * 80)
    print("📋 数据样本 (前5行)")
    print("=" * 80)
    print()
    sample_cols = ['target_drug', 'seriousnessdeath', 'seriousnesshospitalization', 
                   'patientsex', 'patientonsetage', 'num_drugs', 'num_reactions']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head())
    print()
    
    print("=" * 80)
    print("🎯 下一步")
    print("=" * 80)
    print()
    print("数据已准备好，可以开始后续步骤:")
    print("  1. 运行: python step2_inspect_data.py")
    print("  2. 或者手动检查数据: open task5_severity_prediction_data.csv")
    print()
    print("💡 提示: 后续步骤会自动识别并使用这个数据文件")
    print()

if __name__ == "__main__":
    main()

