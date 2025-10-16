#!/usr/bin/env python3
"""
Task 5 - 步骤 4/7: 训练模型（简化版）

训练多个机器学习模型并比较性能
"""

import sys
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Task 5 - 步骤 4/7: 训练模型")
print("=" * 80)
print()

# 检查输入文件
required_files = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"❌ 错误: 缺少必要文件:")
    for f in missing_files:
        print(f"  - {f}")
    print()
    print("请先运行: python step3_preprocess_data.py")
    sys.exit(1)

print("✅ 找到所有必要文件")
print()

# 加载数据
print("📂 加载训练和测试数据...")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"训练集: {len(X_train)} 样本, {len(X_train.columns)} 特征")
print(f"测试集: {len(X_test)} 样本")
print()

# 导入模型
print("🔧 初始化模型...")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 检查XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("✅ XGBoost 可用")
except:
    HAS_XGBOOST = False
    print("⚠️  XGBoost 不可用（跳过）")

print()

# 训练模型
print("=" * 80)
print("🚀 开始训练模型")
print("=" * 80)
print()

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

if HAS_XGBOOST:
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')

print(f"将训练 {len(models)} 个模型:")
for i, name in enumerate(models.keys(), 1):
    print(f"  {i}. {name}")
print()
print("⏱️  预计时间: 5-10分钟")
print()

# 训练并评估
results = {}

for i, (name, model) in enumerate(models.items(), 1):
    print(f"[{i}/{len(models)}] 训练 {name}...")
    
    try:
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"   ✅ 完成 - 准确度: {accuracy:.4f}")
        
        # 保存模型
        model_filename = f"trained_model_{name.lower().replace(' ', '_')}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
    except Exception as e:
        print(f"   ❌ 失败: {str(e)[:50]}")
        results[name] = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'roc_auc': 0
        }

print()
print("✅ 模型训练完成")
print()

# 显示结果
print("=" * 80)
print("📊 模型性能比较")
print("=" * 80)
print()

# 创建结果表格
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

# 按准确度排序
results_df = results_df.sort_values('accuracy', ascending=False)

print(results_df.to_string())
print()

# 找出最佳模型
best_model_name = results_df['accuracy'].idxmax()
best_accuracy = results_df.loc[best_model_name, 'accuracy']

print(f"🏆 最佳模型: {best_model_name}")
print(f"   准确度: {best_accuracy:.4f}")
print(f"   F1分数: {results_df.loc[best_model_name, 'f1']:.4f}")
print(f"   ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
print()

# 模型解释
print("=" * 80)
print("📖 指标说明")
print("=" * 80)
print()

print("Accuracy (准确度): 预测正确的比例")
print("Precision (精确率): 预测为阳性中实际为阳性的比例")
print("Recall (召回率): 实际阳性中被正确预测的比例")
print("F1-Score: 精确率和召回率的调和平均")
print("ROC-AUC: 模型区分能力的综合指标 (越接近1越好)")
print()

# 分析结果
print("=" * 80)
print("💡 结果分析")
print("=" * 80)
print()

print("模型性能评估:")
if best_accuracy >= 0.8:
    print("  🌟 优秀: 模型性能很好")
elif best_accuracy >= 0.7:
    print("  ✅ 良好: 模型性能不错")
elif best_accuracy >= 0.6:
    print("  ⚠️  一般: 模型性能中等")
else:
    print("  ❌ 较差: 可能需要更多特征工程")
print()

# 特征重要性（如果是树模型）
print("=" * 80)
print("🔍 特征重要性（最佳模型）")
print("=" * 80)
print()

best_model_file = f"trained_model_{best_model_name.lower().replace(' ', '_')}.pkl"
if os.path.exists(best_model_file):
    with open(best_model_file, 'rb') as f:
        best_model = pickle.load(f)
    
    if hasattr(best_model, 'feature_importances_'):
        # 树模型
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 重要特征:")
        print(feature_importance.head(10).to_string(index=False))
        
        # 保存特征重要性
        feature_importance.to_csv("feature_importance.csv", index=False)
        print()
        print("✅ 已保存: feature_importance.csv")
        
    elif hasattr(best_model, 'coef_'):
        # 线性模型
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': abs(best_model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        print("Top 10 重要特征:")
        print(feature_importance.head(10).to_string(index=False))
        
        # 保存特征重要性
        feature_importance.to_csv("feature_importance.csv", index=False)
        print()
        print("✅ 已保存: feature_importance.csv")
    else:
        print("⚠️  该模型不支持直接提取特征重要性")
else:
    print("⚠️  无法加载最佳模型文件")

print()

# 保存结果
print("💾 保存结果...")
results_df.to_csv("model_comparison.csv")
print("✅ 已保存: model_comparison.csv")
print()

# 总结
print("=" * 80)
print("✅ 步骤4完成 - 模型训练完毕")
print("=" * 80)
print()

print("📁 生成的文件:")
print("  1. model_comparison.csv - 模型性能比较")
print("  2. trained_model_*.pkl - 训练好的模型文件")
if os.path.exists("feature_importance.csv"):
    print("  3. feature_importance.csv - 特征重要性")
print()

print("🎯 下一步:")
print("  运行: python step5_analyze_features.py")
print("  作用: 详细分析特征重要性")
print()

print("💡 提示:")
print(f"  您可以查看模型性能: open model_comparison.csv")
print()

