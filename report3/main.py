import sys
import io
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

# Windows 控制台 UTF-8 编码设置
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 1. 加载数据
# ============================================================
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("=" * 60)
print("California Housing Dataset")
print("=" * 60)
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {list(data.feature_names)}")

# ============================================================
# 2. 数据预处理: 剔除 AveRooms > 20 的异常样本
# ============================================================
outlier_mask = X['AveRooms'] <= 20
X_clean = X[outlier_mask].reset_index(drop=True)
y_clean = y[outlier_mask.to_numpy()]

n_outliers = (~outlier_mask).sum()
print(f"\nRemoved outliers (AveRooms > 20): {n_outliers} samples")
print(f"Remaining valid samples: {len(X_clean)}")

# ============================================================
# 3. 数据集划分 (7:3)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================
# 4. 模型训练
# ============================================================

common_params = {
    'boosting_type': 'rf',
    'n_estimators': 100,
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': 42,
    'bagging_freq': 1,
    'bagging_fraction': 0.632,
    'verbose': -1,
}

# 基线模型: 无单调性约束
print("\nTraining baseline model (no monotonicity constraint)...")
model_baseline = lgb.LGBMRegressor(**common_params)
model_baseline.fit(X_train, y_train)

# 实验模型: 对 AveRooms (索引2) 施加单调递增约束
# 特征顺序: MedInc(0), HouseAge(1), AveRooms(2), AveBedrms(3),
#            Population(4), AveOccup(5), Latitude(6), Longitude(7)
monotone_constraints = [0, 0, 1, 0, 0, 0, 0, 0]
print("Training constrained model (monotonicity on AveRooms)...")
model_constrained = lgb.LGBMRegressor(
    **common_params,
    monotone_constraints=monotone_constraints,
)
model_constrained.fit(X_train, y_train)

# ============================================================
# 5. 预测与精度评估
# ============================================================
y_pred_baseline = model_baseline.predict(X_test)
y_pred_constrained = model_constrained.predict(X_test)

def evaluate(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
    }

results_baseline = evaluate(y_test, y_pred_baseline)
results_constrained = evaluate(y_test, y_pred_constrained)

print("\n" + "=" * 60)
print("Prediction Accuracy Comparison")
print("=" * 60)
print(f"{'Metric':<10} {'Baseline RF':<18} {'Constrained RF':<18}")
print("-" * 48)
for metric in ['R2', 'MAE', 'RMSE']:
    print(f"{metric:<10} {results_baseline[metric]:<18.4f} {results_constrained[metric]:<18.4f}")

# ============================================================
# 6. 单调性验证: MVR 计算
# ============================================================
X_mean = X_test.mean().values.reshape(1, -1)
ave_rooms_range = np.linspace(1, 10, 100)

def compute_mvr(model, X_mean, ave_rooms_range):
    """Calculate Monotonicity Violation Rate"""
    X_sampled = np.tile(X_mean, (len(ave_rooms_range), 1))
    X_sampled[:, 2] = ave_rooms_range  # AveRooms column
    predictions = model.predict(X_sampled)
    diffs = np.diff(predictions)
    violations = np.sum(diffs < 0)
    total_changes = len(diffs)
    mvr = violations / total_changes
    return predictions, mvr

pred_baseline_mono, mvr_baseline = compute_mvr(model_baseline, X_mean, ave_rooms_range)
pred_constrained_mono, mvr_constrained = compute_mvr(model_constrained, X_mean, ave_rooms_range)

print("\n" + "=" * 60)
print("Monotonicity Verification")
print("=" * 60)
print(f"{'Model':<22} {'MVR':<12} {'Strictly Increasing':<20}")
print("-" * 54)
print(f"{'Baseline RF':<22} {mvr_baseline:<12.2%} {'No':<20}")
print(f"{'Constrained RF':<22} {mvr_constrained:<12.2%} {'Yes' if mvr_constrained == 0 else 'No':<20}")

# ============================================================
# 7. 保存数据供画图使用
# ============================================================
np.savez('report3/monotonicity_data.npz',
         ave_rooms_range=ave_rooms_range,
         pred_baseline=pred_baseline_mono,
         pred_constrained=pred_constrained_mono,
         mvr_baseline=mvr_baseline,
         mvr_constrained=mvr_constrained,
         y_test=y_test,
         y_pred_baseline=y_pred_baseline,
         y_pred_constrained=y_pred_constrained)

print("\nData saved to report3/monotonicity_data.npz")
print("Done!")
