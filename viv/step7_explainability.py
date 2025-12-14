#!/usr/bin/env python3
"""
Task 5 - 步骤 7/7: 模型可解释性分析

使用SHAP和LIME分析模型预测的可解释性
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys

print("=" * 80)
print("Task 5 - 步骤 7/7: 模型可解释性分析")
print("=" * 80)
print()

# 检查必要文件
print("🔍 检查必要文件...")
required_files = {
    'X_train.csv': '训练特征',
    'X_test.csv': '测试特征',
    'y_test.csv': '测试标签',
    'trained_model_gradient_boosting.pkl': '最佳模型'
}

missing = []
for file, desc in required_files.items():
    if os.path.exists(file):
        print(f"  ✅ {desc}: {file}")
    else:
        print(f"  ❌ 缺失 {desc}: {file}")
        missing.append(file)

if missing:
    print()
    print("❌ 错误: 缺少必要文件")
    print("   请先运行: python step3_preprocess_data.py")
    print("   然后运行: python step4_train_models.py")
    sys.exit(1)

print()

# 检查SHAP是否可用
try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP 已安装")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP 未安装")
    print("   安装命令: pip install shap")

# 检查LIME是否可用
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
    print("✅ LIME 已安装")
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME 未安装")
    print("   安装命令: pip install lime")

print()

if not SHAP_AVAILABLE and not LIME_AVAILABLE:
    print("❌ 错误: SHAP和LIME都未安装")
    print()
    print("请安装:")
    print("  pip install shap lime")
    sys.exit(1)

# 加载数据
print("📂 加载数据...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# 重置索引以避免索引不匹配
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.squeeze().reset_index(drop=True)

print(f"✅ 训练集: {len(X_train)} 样本, {len(X_train.columns)} 特征")
print(f"✅ 测试集: {len(X_test)} 样本")
print()

# 加载最佳模型（Gradient Boosting）
print("🤖 加载最佳模型...")
with open("trained_model_gradient_boosting.pkl", 'rb') as f:
    model = pickle.load(f)
print("✅ 模型加载成功: Gradient Boosting")
print()

# ============================================================================
# SHAP Analysis
# ============================================================================

if SHAP_AVAILABLE:
    print("=" * 80)
    print("📊 SHAP 可解释性分析")
    print("=" * 80)
    print()
    
    import matplotlib.pyplot as plt
    
    # 限制样本数以提高速度
    MAX_SAMPLES = 500
    sample_indices = np.random.RandomState(42).choice(len(X_test), 
                                                      size=min(MAX_SAMPLES, len(X_test)), 
                                                      replace=False)
    X_test_sample = X_test.iloc[sample_indices].reset_index(drop=True)
    y_test_sample = y_test.iloc[sample_indices].reset_index(drop=True)
    
    print(f"使用 {len(X_test_sample)} 个测试样本进行SHAP分析")
    print("⏱️  预计时间: 2-5分钟...")
    print()
    
    try:
        # 1. 初始化SHAP Explainer
        print("[1/5] 初始化SHAP解释器...")
        explainer = shap.TreeExplainer(model)
        print("      ✅ 完成")
        print()
        
        # 2. 计算SHAP值
        print("[2/5] 计算SHAP值...")
        shap_values = explainer.shap_values(X_test_sample)
        print("      ✅ 完成")
        print()
        
        # 3. SHAP摘要图（散点图）
        print("[3/5] 生成SHAP摘要图...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, max_display=15, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("      ✅ 已保存: shap_summary_plot.png")
        print()
        
        # 4. SHAP条形图（平均绝对值）
        print("[4/5] 生成SHAP条形图...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", 
                         max_display=15, show=False)
        plt.title("SHAP Feature Importance (Mean |SHAP|)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("      ✅ 已保存: shap_bar_plot.png")
        print()
        
        # 5. 个体预测解释（选择几个有代表性的案例）
        print("[5/5] 生成个体预测解释...")
        
        # 找一个死亡案例和一个存活案例
        death_mask = y_test_sample == 1
        survival_mask = y_test_sample == 0
        
        sample_cases = []
        if death_mask.any():
            death_pos = np.where(death_mask)[0][0]
            sample_cases.append(('death', death_pos))
        if survival_mask.any():
            survival_pos = np.where(survival_mask)[0][0]
            sample_cases.append(('survival', survival_pos))
        
        for case_type, sample_pos in sample_cases:
            
            # 创建SHAP解释对象
            shap_explanation = shap.Explanation(
                values=shap_values[sample_pos],
                base_values=explainer.expected_value,
                data=X_test_sample.iloc[sample_pos].values,
                feature_names=X_test_sample.columns.tolist()
            )
            
            # 瀑布图
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(shap_explanation, max_display=15, show=False)
            plt.title(f"SHAP Waterfall Plot - {case_type.title()} Case (Sample {sample_pos})", 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"shap_waterfall_{case_type}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✅ 已保存: shap_waterfall_{case_type}.png")
        
        print()
        
        # 保存SHAP值到CSV
        shap_df = pd.DataFrame(
            shap_values,
            columns=[f"shap_{col}" for col in X_test_sample.columns]
        )
        shap_df['y_true'] = y_test_sample.values
        shap_df.to_csv("shap_values.csv", index=False)
        print("✅ 已保存: shap_values.csv")
        print()
        
        # 计算全局特征重要性
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X_test_sample.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("📊 Top 10 全局特征重要性 (Mean |SHAP|):")
        print()
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:25s}: {row['mean_abs_shap']:8.4f}")
        print()
        
        importance_df.to_csv("shap_feature_importance.csv", index=False)
        print("✅ 已保存: shap_feature_importance.csv")
        print()
        
        print("✅ SHAP分析完成")
        print()
        
    except Exception as e:
        print(f"❌ SHAP分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

# ============================================================================
# LIME Analysis
# ============================================================================

if LIME_AVAILABLE:
    print("=" * 80)
    print("📊 LIME 局部可解释性分析")
    print("=" * 80)
    print()
    
    import matplotlib.pyplot as plt
    from lime.lime_tabular import LimeTabularExplainer
    
    try:
        # 初始化LIME解释器
        print("[1/3] 初始化LIME解释器...")
        lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Survival', 'Death'],
            mode='classification',
            random_state=42
        )
        print("      ✅ 完成")
        print()
        
        # 选择代表性案例
        print("[2/3] 生成LIME解释...")
        
        # 找一个死亡案例和一个存活案例
        death_mask = y_test == 1
        survival_mask = y_test == 0
        
        cases = []
        if death_mask.any():
            death_pos = np.where(death_mask)[0][0]
            cases.append(('death', death_pos))
        if survival_mask.any():
            survival_pos = np.where(survival_mask)[0][0]
            cases.append(('survival', survival_pos))
        
        for case_type, test_pos in cases:
            instance = X_test.iloc[test_pos].values
            
            # 生成LIME解释
            explanation = lime_explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=10
            )
            
            # 保存图表
            fig = explanation.as_pyplot_figure()
            plt.title(f"LIME Explanation - {case_type.title()} Case", 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"lime_explanation_{case_type}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✅ 已保存: lime_explanation_{case_type}.png")
            
            # 保存文本解释
            with open(f"lime_explanation_{case_type}.txt", 'w') as f:
                f.write(f"LIME Explanation - {case_type.title()} Case\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Prediction: {model.predict([instance])[0]}\n")
                f.write(f"Prediction Probability: {model.predict_proba([instance])[0]}\n\n")
                f.write("Feature Contributions:\n")
                f.write("-" * 60 + "\n")
                for feature, weight in explanation.as_list():
                    f.write(f"{feature:40s}: {weight:10.4f}\n")
            
            print(f"      ✅ 已保存: lime_explanation_{case_type}.txt")
        
        print()
        print("[3/3] LIME分析完成")
        print()
        
    except Exception as e:
        print(f"❌ LIME分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

# ============================================================================
# 总结
# ============================================================================

print("=" * 80)
print("✅ 步骤7完成 - 模型可解释性分析完毕")
print("=" * 80)
print()

print("📁 生成的文件:")
if SHAP_AVAILABLE:
    print("  SHAP分析:")
    print("    1. shap_summary_plot.png - SHAP特征重要性摘要")
    print("    2. shap_bar_plot.png - SHAP特征重要性条形图")
    print("    3. shap_waterfall_death.png - 死亡案例解释")
    print("    4. shap_waterfall_survival.png - 存活案例解释")
    print("    5. shap_values.csv - 完整SHAP值")
    print("    6. shap_feature_importance.csv - 全局特征重要性")
    print()

if LIME_AVAILABLE:
    print("  LIME分析:")
    print("    7. lime_explanation_death.png - 死亡案例解释")
    print("    8. lime_explanation_survival.png - 存活案例解释")
    print("    9. lime_explanation_death.txt - 死亡案例文本解释")
    print("   10. lime_explanation_survival.txt - 存活案例文本解释")
    print()

print("🎯 关键发现:")
if SHAP_AVAILABLE:
    print("  - SHAP分析揭示了模型决策的全局和局部模式")
    print("  - 特征重要性排序帮助理解哪些因素影响死亡风险")
if LIME_AVAILABLE:
    print("  - LIME提供了单个预测的可解释性")
    print("  - 可以向临床医生解释特定患者的风险因素")
print()

print("💡 可解释性意义:")
print("  - 提高模型透明度，增强临床可信度")
print("  - 识别关键风险因素，指导临床决策")
print("  - 满足监管要求（FDA AI/ML指南）")
print("  - 促进模型改进和特征工程")
print()

print("🎓 项目完成!")
print("  所有7个步骤已完成:")
print("    ✅ 步骤1: 数据提取")
print("    ✅ 步骤2: 数据检查")
print("    ✅ 步骤3: 数据预处理")
print("    ✅ 步骤4: 模型训练")
print("    ✅ 步骤5: 特征分析")
print("    ✅ 步骤6: 结果可视化")
print("    ✅ 步骤7: 可解释性分析")
print()

