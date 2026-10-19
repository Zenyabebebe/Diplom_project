# 02_2_deviation_model.py
# Модель предсказания отклонения Early/late
# Финальный срок = PD × (1 + Early_late_pred)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === 1. ПУТИ ===
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "group_a.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# === 2. ЗАГРУЗКА ===
df = pd.read_csv(INPUT_FILE)
print("=" * 60)
print("МОДЕЛЬ ОТКЛОНЕНИЯ (Early/late)")
print("=" * 60)
print(f"Проектов: {len(df)}")

# === 3. ПЕРЕИМЕНОВАНИЕ КОЛОНОК ===
cols = list(df.columns)

sensitivity_groups = [
    "CI", "SI", "SSI",
    "CRI_r", "CRI_rho", "CRI_tau",
    "CRI_r_cost", "CRI_rho_cost", "CRI_tau_cost",
    "CRI_r_res", "CRI_rho_res", "CRI_tau_res",
]
perf_groups = ["CV", "SV", "SV_t", "CPI", "SPI", "SPI_t", "p_factor"]

start_idx = cols.index("avg")
rename_map = {}
idx = start_idx
for group in sensitivity_groups:
    for metric_suffix, metric_name in zip(
        ["avg", "std. dev.", "skew"], ["avg", "std", "skew"]
    ):
        if idx < len(cols):
            rename_map[cols[idx]] = f"{group}_{metric_name}"
            idx += 1

perf_start_idx = start_idx + 36
idx = perf_start_idx
for group in perf_groups:
    for metric_suffix, metric_name in zip(
        ["avg", "std. dev.", "final"], ["avg", "std", "final"]
    ):
        if idx < len(cols):
            old = cols[idx]
            if old in ["Duration", "Cost"]:
                break
            rename_map[old] = f"{group}_{metric_name}"
            idx += 1

df = df.rename(columns=rename_map)

# === 4. УНИФИКАЦИЯ SECTOR ===
df["Sector"] = df["Sector"].astype(str).str.lower().str.strip()
sector_dummies = pd.get_dummies(df["Sector"], prefix="sector")
df = pd.concat([df, sector_dummies], axis=1)

# === 5. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ===
df["PD_num"] = pd.to_numeric(df["PD (days)"], errors="coerce")
df["Duration_num"] = pd.to_numeric(df["Duration"], errors="coerce")

# Пересчитываем Early/late сами
df["Early_late_calc"] = (df["Duration_num"] - df["PD_num"]) / df["PD_num"]

# Сравниваем с готовой колонкой
df["Early_late_given"] = pd.to_numeric(df["Early/late"], errors="coerce")

print("\n" + "=" * 60)
print("ПРОВЕРКА EARLY/LATE")
print("=" * 60)
print("Первые 10 значений (calc vs given):")
print(df[["Code", "PD_num", "Duration_num", "Early_late_calc", "Early_late_given"]].head(10).to_string(index=False))

diff = (df["Early_late_calc"] - df["Early_late_given"]).abs()
print(f"\nМаксимальное расхождение: {diff.max():.6f}")
print(f"Среднее расхождение: {diff.mean():.6f}")

# Используем пересчитанную
y_dev = df["Early_late_calc"].copy()

# Убираем строки, где Early/late не посчитан
valid_mask = y_dev.notna() & df["PD_num"].notna() & df["Duration_num"].notna()
df = df[valid_mask].reset_index(drop=True)
y_dev = y_dev[valid_mask].reset_index(drop=True)

print(f"\nПроектов после фильтрации: {len(df)}")
print(f"Early/late: mean={y_dev.mean():.3f}, std={y_dev.std():.3f}, min={y_dev.min():.3f}, max={y_dev.max():.3f}")

# === 6. ПРИЗНАКИ (БЕЗ PD И EARLY/LATE) ===
base_features = [
    "# activities", "BAC",
    "SP", "AD", "LA", "TF", "RI",
]
sensitivity_features = [c for c in df.columns if any(
    c.startswith(g + "_") for g in sensitivity_groups + perf_groups
)]
sector_features = [c for c in df.columns if c.startswith("sector_")]

all_features = base_features + sensitivity_features + sector_features
all_features = [f for f in all_features if f in df.columns]

print(f"\nПризнаков (без PD и Early/late): {len(all_features)}")

# === 7. ПОДГОТОВКА X, y ===
X = df[all_features].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Удаляем признаки с >50% пропусков
missing_pct = X.isna().mean() * 100
drop_cols = missing_pct[missing_pct > 50].index.tolist()
if drop_cols:
    print(f"\nУдаляем {len(drop_cols)} признаков с >50% пропусков:")
    for c in drop_cols:
        print(f"  - {c}")
    X = X.drop(columns=drop_cols)
    all_features = [f for f in all_features if f not in drop_cols]

X = X.fillna(X.median(numeric_only=True))
print(f"Итоговое число признаков: {X.shape[1]}")

# Сохраняем PD и Duration для восстановления
PD = df["PD_num"].values
Duration_real = df["Duration_num"].values

# === 8. 5-FOLD CV С ТРЕМЯ МОДЕЛЯМИ ===
print("\n" + "=" * 60)
print("5-FOLD CV: СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Массивы для OOF-предсказаний
oof_baseline_zero = np.zeros(len(X))
oof_baseline_mean = np.zeros(len(X))
oof_rf = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

    # Baseline 0
    oof_baseline_zero[val_idx] = 0.0

    # Baseline mean
    mean_train = y_train.mean()
    oof_baseline_mean[val_idx] = mean_train

    # RF
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    oof_rf[val_idx] = model.predict(X_val)

    # Метрики по отклонению
    mae_b0 = mean_absolute_error(y_val, oof_baseline_zero[val_idx])
    mae_bm = mean_absolute_error(y_val, oof_baseline_mean[val_idx])
    mae_rf = mean_absolute_error(y_val, oof_rf[val_idx])
    print(f"Fold {fold}: MAE (откл) — B0={mae_b0:.3f}, Bmean={mae_bm:.3f}, RF={mae_rf:.3f}")

# === 9. ВОССТАНОВЛЕНИЕ DURATION ===
# Duration = PD × (1 + Early_late_pred)
dur_baseline_zero = PD * (1 + oof_baseline_zero)
dur_baseline_mean = PD * (1 + oof_baseline_mean)
dur_rf = PD * (1 + oof_rf)

# === 10. МЕТРИКИ ПО ОТКЛОНЕНИЮ ===
print("\n" + "=" * 60)
print("МЕТРИКИ ПО ОТКЛОНЕНИЮ (Early/late)")
print("=" * 60)

def eval_dev(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs(y_true - y_pred)) * 100  # MAE в процентных пунктах
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.4f} ({mae*100:.2f} п.п.)")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

eval_dev(y_dev.values, oof_baseline_zero, "Baseline 0 (Early_late = 0)")
eval_dev(y_dev.values, oof_baseline_mean, "Baseline mean (Early_late = mean)")
eval_dev(y_dev.values, oof_rf, "Random Forest")

# === 11. МЕТРИКИ ПО DURATION ===
print("\n" + "=" * 60)
print("МЕТРИКИ ПО DURATION (дни)")
print("=" * 60)

def eval_dur(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.1f} дней")
    print(f"  RMSE: {rmse:.1f} дней")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R²:   {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

results = {}
results["Baseline PD"] = eval_dur(Duration_real, PD, "Baseline: Duration = PD")
results["Baseline 0"] = eval_dur(Duration_real, dur_baseline_zero, "Baseline 0 (Early_late = 0)")
results["Baseline mean"] = eval_dur(Duration_real, dur_baseline_mean, "Baseline mean")
results["RF"] = eval_dur(Duration_real, dur_rf, "RF (Early/late)")

# === 12. ИТОГОВАЯ ТАБЛИЦА ===
print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА (Duration)")
print("=" * 60)
summary = pd.DataFrame(results).T
summary = summary[["MAE", "RMSE", "MAPE", "R2"]]
summary = summary.round(3)
print(summary.to_string())

# === 13. ФИНАЛЬНАЯ МОДЕЛЬ ===
print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ МОДЕЛЬ НА ВСЕЙ ВЫБОРКЕ")
print("=" * 60)

final_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y_dev)

joblib.dump({
    "model": final_model,
    "features": all_features,
    "target": "Early_late",
}, MODEL_DIR / "rf_deviation_model.pkl")
print(f"✅ Модель сохранена: {MODEL_DIR / 'rf_deviation_model.pkl'}")

# === 14. ВАЖНОСТЬ ПРИЗНАКОВ ===
print("\n" + "=" * 60)
print("ТОП-25 ВАЖНЫХ ПРИЗНАКОВ (для Early/late)")
print("=" * 60)

importances = pd.Series(final_model.feature_importances_, index=all_features)
importances = importances.sort_values(ascending=False)
print(importances.head(25).to_string())

# Сохраняем
importances_df = pd.DataFrame({
    "Признак": importances.index,
    "Важность": importances.values,
})
importances_df.to_csv(OUTPUT_DIR / "feature_importance_deviation.csv", index=False)

# === 15. ГРАФИК ВАЖНОСТИ ===
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = importances.head(20)
ax.barh(range(len(top_20)), top_20.values, color="steelblue")
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20.index)
ax.invert_yaxis()
ax.set_xlabel("Важность")
ax.set_title("Топ-20 важных признаков для предсказания Early/late")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_deviation.png", dpi=150)
plt.close()
print(f"\n✅ График: {OUTPUT_DIR / 'feature_importance_deviation.png'}")

# === 16. РАСПРЕДЕЛЕНИЕ ОТКЛОНЕНИЙ ===
print("\n" + "=" * 60)
print("РАСПРЕДЕЛЕНИЕ ОТКЛОНЕНИЙ")
print("=" * 60)
print(f"Реальные Early/late:")
print(f"  Среднее: {y_dev.mean():+.3f}")
print(f"  Медиана: {y_dev.median():+.3f}")
print(f"  Std:     {y_dev.std():.3f}")
print(f"  P10:     {y_dev.quantile(0.10):+.3f}")
print(f"  P90:     {y_dev.quantile(0.90):+.3f}")

print(f"\nПредсказанные RF Early/late:")
print(f"  Среднее: {oof_rf.mean():+.3f}")
print(f"  Медиана: {np.median(oof_rf):+.3f}")
print(f"  Std:     {oof_rf.std():.3f}")

# === 17. СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ ===
pred_df = pd.DataFrame({
    "Code": df["Code"],
    "Sector": df["Sector"],
    "PD": PD,
    "Duration_real": Duration_real,
    "Early_late_real": y_dev.values,
    "Early_late_rf_pred": oof_rf,
    "Duration_baseline_PD": PD,
    "Duration_rf_pred": dur_rf,
})
pred_df.to_csv(OUTPUT_DIR / "predictions_deviation.csv", index=False)
print(f"\n✅ Предсказания: {OUTPUT_DIR / 'predictions_deviation.csv'}")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)