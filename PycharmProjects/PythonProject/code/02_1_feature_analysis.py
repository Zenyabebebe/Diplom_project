# 02_1_feature_analysis.py
# Анализ важности признаков БЕЗ PD
# Цель: понять, сколько информации несут остальные признаки

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === 1. ПУТИ ===
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "group_a.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# === 2. ЗАГРУЗКА ===
df = pd.read_csv(INPUT_FILE)
print("=" * 60)
print("АНАЛИЗ ПРИЗНАКОВ БЕЗ PD")
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

# === 5. ПРИЗНАКИ БЕЗ PD ===
base_features = [
    "# activities", "BAC",
    "SP", "AD", "LA", "TF", "RI",
]
sensitivity_features = [c for c in df.columns if any(
    c.startswith(g + "_") for g in sensitivity_groups + perf_groups
)]
sector_features = list(sector_dummies.columns)

all_features = base_features + sensitivity_features + sector_features
all_features = [f for f in all_features if f in df.columns]

print(f"\nПризнаков (без PD): {len(all_features)}")

# === 6. ПОДГОТОВКА X, y ===
X = df[all_features].copy()
y = df["Duration"].copy()

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

# Заполняем медианой
X = X.fillna(X.median(numeric_only=True))

print(f"Итоговое число признаков: {X.shape[1]}")

# === 7. 5-FOLD CV ===
print("\n" + "=" * 60)
print("5-FOLD CV (БЕЗ PD)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    oof_pred[val_idx] = model.predict(X_val)

    mae = mean_absolute_error(y_val, oof_pred[val_idx])
    print(f"Fold {fold}: MAE = {mae:.1f} дней")

# === 8. ИТОГОВЫЕ МЕТРИКИ ===
mae = mean_absolute_error(y, oof_pred)
rmse = np.sqrt(mean_squared_error(y, oof_pred))
r2 = r2_score(y, oof_pred)
mape = np.mean(np.abs((y - oof_pred) / y)) * 100

print(f"\nИТОГИ (RF без PD, целевая Duration):")
print(f"  MAE:  {mae:.1f} дней")
print(f"  RMSE: {rmse:.1f} дней")
print(f"  MAPE: {mape:.1f}%")
print(f"  R²:   {r2:.3f}")

# Сравнение с baseline
pd_values = df["PD (days)"].astype(float)
baseline_mae = mean_absolute_error(y, pd_values)
print(f"\nСравнение с baseline (Duration = PD):")
print(f"  Baseline MAE: {baseline_mae:.1f} дней")
print(f"  RF без PD MAE: {mae:.1f} дней")
print(f"  Разница: {mae - baseline_mae:+.1f} дней")

# === 9. ОБУЧАЕМ ФИНАЛЬНУЮ МОДЕЛЬ ===
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ (без PD)")
print("=" * 60)

final_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y)

# === 10. ВАЖНОСТЬ ПРИЗНАКОВ ===
print("\n" + "=" * 60)
print("ТОП-30 ВАЖНЫХ ПРИЗНАКОВ (без PD)")
print("=" * 60)

importances = pd.Series(final_model.feature_importances_, index=all_features)
importances = importances.sort_values(ascending=False)

# Вывод компактно, группами
print(importances.head(30).to_string())

# === 11. ГРАФИК ===
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = importances.head(20)
ax.barh(range(len(top_20)), top_20.values)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20.index)
ax.invert_yaxis()
ax.set_xlabel("Важность")
ax.set_title("Топ-20 важных признаков (без PD)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_no_pd.png", dpi=150)
plt.close()
print(f"\n✅ График сохранён: {OUTPUT_DIR / 'feature_importance_no_pd.png'}")

# === 12. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
importances_df = pd.DataFrame({
    "Признак": importances.index,
    "Важность": importances.values,
})
importances_df.to_csv(OUTPUT_DIR / "feature_importance_no_pd.csv", index=False)

results = pd.DataFrame({
    "Code": df["Code"],
    "Sector": df["Sector"],
    "Duration_real": y,
    "Duration_pred_no_pd": oof_pred,
    "Duration_baseline_PD": pd_values,
})
results.to_csv(OUTPUT_DIR / "predictions_no_pd.csv", index=False)

print(f"✅ Важность признаков сохранена: {OUTPUT_DIR / 'feature_importance_no_pd.csv'}")
print(f"✅ Предсказания сохранены: {OUTPUT_DIR / 'predictions_no_pd.csv'}")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)