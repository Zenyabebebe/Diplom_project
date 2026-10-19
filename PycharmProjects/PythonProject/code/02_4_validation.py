# 02_4_validation.py
# Валидация модели:
# 1. Leave-One-Out CV
# 2. Применение к группе Б (проекты без Duration)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === 1. ПУТИ ===
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "group_a.csv"
GROUP_B_FILE = BASE_DIR / "output" / "construction_all.csv"
OUTPUT_DIR = BASE_DIR / "output"

# === 2. ЗАГРУЗКА ===
df = pd.read_csv(INPUT_FILE)
print("=" * 60)
print("ВАЛИДАЦИЯ МОДЕЛИ")
print("=" * 60)
print(f"Группа А: {len(df)} проектов")

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

# === 4. SECTOR ===
df["Sector"] = df["Sector"].astype(str).str.lower().str.strip()
sector_dummies = pd.get_dummies(df["Sector"], prefix="sector")
df = pd.concat([df, sector_dummies], axis=1)

# === 5. ЦЕЛЕВАЯ ===
df["PD_num"] = pd.to_numeric(df["PD (days)"], errors="coerce")
df["Duration_num"] = pd.to_numeric(df["Duration"], errors="coerce")
df["Early_late_calc"] = (df["Duration_num"] - df["PD_num"]) / df["PD_num"]

valid_mask = df["Early_late_calc"].notna() & df["PD_num"].notna() & df["Duration_num"].notna()
df = df[valid_mask].reset_index(drop=True)
y_dev = df["Early_late_calc"].copy()
PD = df["PD_num"].values
Duration_real = df["Duration_num"].values

print(f"Проектов после фильтрации: {len(df)}")

# === 6. ПРИЗНАКИ ===
base_features = ["# activities", "BAC", "SP", "AD", "LA", "TF", "RI"]
sensitivity_features = [c for c in df.columns if any(
    c.startswith(g + "_") for g in sensitivity_groups + perf_groups
)]
sector_features = [c for c in df.columns if c.startswith("sector_")]

all_features = base_features + sensitivity_features + sector_features
all_features = [f for f in all_features if f in df.columns]

X = df[all_features].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Убираем >50% пропусков
missing_pct = X.isna().mean() * 100
drop_cols = missing_pct[missing_pct > 50].index.tolist()
if drop_cols:
    X = X.drop(columns=drop_cols)
    all_features = [f for f in all_features if f not in drop_cols]

X = X.fillna(X.median(numeric_only=True))
print(f"Признаков: {X.shape[1]}")

# === 7. LEAVE-ONE-OUT CV ===
print("\n" + "=" * 60)
print("LEAVE-ONE-OUT CV")
print("=" * 60)
print("(Обучаем 116 раз на 115 проектах, предсказываем 1)")

loo = LeaveOneOut()
loo_pred = np.zeros(len(X))

for i, (train_idx, val_idx) in enumerate(loo.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train = y_dev.iloc[train_idx]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    loo_pred[val_idx] = model.predict(X_val)

    if (i + 1) % 20 == 0:
        print(f"  Обработано: {i + 1}/{len(X)}")

# Метрики LOO
dur_loo = PD * (1 + loo_pred)

mae_loo_dev = mean_absolute_error(y_dev, loo_pred)
mae_loo_dur = mean_absolute_error(Duration_real, dur_loo)
rmse_loo_dur = np.sqrt(mean_squared_error(Duration_real, dur_loo))
mape_loo_dur = np.mean(np.abs((Duration_real - dur_loo) / Duration_real)) * 100
r2_loo_dur = r2_score(Duration_real, dur_loo)

print("\nМЕТРИКИ LOO (по отклонению):")
print(f"  MAE:  {mae_loo_dev:.4f} ({mae_loo_dev*100:.2f} п.п.)")

print("\nМЕТРИКИ LOO (по Duration):")
print(f"  MAE:  {mae_loo_dur:.1f} дней")
print(f"  RMSE: {rmse_loo_dur:.1f} дней")
print(f"  MAPE: {mape_loo_dur:.1f}%")
print(f"  R²:   {r2_loo_dur:.3f}")

# === 8. СРАВНЕНИЕ 5-FOLD vs LOO ===
print("\n" + "=" * 60)
print("СРАВНЕНИЕ 5-FOLD CV vs LOO CV")
print("=" * 60)

print(f"{'Метрика':<15} {'5-Fold':>12} {'LOO':>12} {'Разница':>12}")
print(f"{'-'*55}")
print(f"{'MAE (дни)':<15} {23.3:>12.1f} {mae_loo_dur:>12.1f} {mae_loo_dur - 23.3:>+12.1f}")
print(f"{'RMSE (дни)':<15} {45.4:>12.1f} {rmse_loo_dur:>12.1f} {rmse_loo_dur - 45.4:>+12.1f}")
print(f"{'MAPE (%)':<15} {21.0:>12.1f} {mape_loo_dur:>12.1f} {mape_loo_dur - 21.0:>+12.1f}")
print(f"{'R²':<15} {0.942:>12.3f} {r2_loo_dur:>12.3f} {r2_loo_dur - 0.942:>+12.3f}")

# === 9. ОБУЧАЕМ ФИНАЛЬНУЮ МОДЕЛЬ НА ВСЕЙ ГРУППЕ А ===
print("\n" + "=" * 60)
print("ПРИМЕНЕНИЕ К ГРУППЕ Б")
print("=" * 60)

final_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y_dev)
print("✅ Финальная модель обучена на группе А")

# === 10. ЗАГРУЗКА ГРУППЫ Б ===
df_all = pd.read_csv(GROUP_B_FILE)
print(f"\nЗагружено {len(df_all)} проектов из construction_all.csv")

# Переименовываем так же
df_all = df_all.rename(columns=rename_map)
df_all["Sector"] = df_all["Sector"].astype(str).str.lower().str.strip()

# Дублируем one-hot
sector_dummies_b = pd.get_dummies(df_all["Sector"], prefix="sector")
df_all = pd.concat([df_all, sector_dummies_b], axis=1)

# Отбираем группу Б: Construction, БЕЗ Duration
df_all["PD_num"] = pd.to_numeric(df_all["PD (days)"], errors="coerce")
df_all["Duration_num"] = pd.to_numeric(df_all["Duration"], errors="coerce")

df_group_b = df_all[
    df_all["Sector"].str.contains("construction", na=False) &
    df_all["Duration_num"].isna() &
    df_all["PD_num"].notna()
].copy()

print(f"Группа Б (без Duration): {len(df_group_b)} проектов")

if len(df_group_b) > 0:
    # Готовим признаки
    for f in all_features:
        if f not in df_group_b.columns:
            df_group_b[f] = 0

    X_b = df_group_b[all_features].copy()
    for col in X_b.columns:
        X_b[col] = pd.to_numeric(X_b[col], errors="coerce")
    X_b = X_b.fillna(X.median(numeric_only=True))

    # Предсказание
    pred_dev_b = final_model.predict(X_b)
    PD_b = df_group_b["PD_num"].values
    pred_dur_b = PD_b * (1 + pred_dev_b)

    # Статистика
    print(f"\nРАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ ГРУППЫ Б:")
    print(f"  Early/late: mean={pred_dev_b.mean():+.3f}, "
          f"std={pred_dev_b.std():.3f}, "
          f"min={pred_dev_b.min():+.3f}, max={pred_dev_b.max():+.3f}")
    print(f"  Duration:   mean={pred_dur_b.mean():.1f}, "
          f"min={pred_dur_b.min():.1f}, max={pred_dur_b.max():.1f}")

    # Проверки
    negative_dur = (pred_dur_b < 0).sum()
    negative_dev = (pred_dev_b < -1).sum()  # отклонение < -100% (абсурд)
    huge_dev = (pred_dev_b > 3).sum()  # отклонение > +300%

    print(f"\nПРОВЕРКИ:")
    print(f"  Отрицательных Duration:  {negative_dur}")
    print(f"  Отклонение < -100%:      {negative_dev}")
    print(f"  Отклонение > +300%:      {huge_dev}")
    print(f"  ✅ Предсказания {'разумны' if (negative_dur + negative_dev + huge_dev) == 0 else '⚠️ требуют анализа'}")

    # Сохраняем
    results_b = pd.DataFrame({
        "Code": df_group_b["Code"],
        "Sector": df_group_b["Sector"],
        "PD": PD_b,
        "Early_late_pred": pred_dev_b,
        "Duration_pred": pred_dur_b,
    })
    results_b.to_csv(OUTPUT_DIR / "predictions_group_b.csv", index=False)
    print(f"\n✅ Предсказания группы Б: {OUTPUT_DIR / 'predictions_group_b.csv'}")

# === 11. ГРАФИК ===
print("\n" + "=" * 60)
print("ГРАФИК: ПРЕДСКАЗАНО vs РЕАЛЬНО (LOO)")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(Duration_real, dur_loo, alpha=0.5, edgecolors="k", linewidths=0.5)
lims = [min(Duration_real.min(), dur_loo.min()), max(Duration_real.max(), dur_loo.max())]
ax.plot(lims, lims, "r--", alpha=0.7, label="Идеальное совпадение")
ax.set_xlabel("Реальный срок (дни)")
ax.set_ylabel("Предсказанный срок (LOO, дни)")
ax.set_title(f"LOO CV: MAE = {mae_loo_dur:.1f} дней, R² = {r2_loo_dur:.3f}")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loo_scatter.png", dpi=150)
plt.close()
print(f"✅ График: {OUTPUT_DIR / 'loo_scatter.png'}")

# === 12. ИТОГОВЫЙ ВЫВОД ===
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)

if abs(mae_loo_dur - 23.3) < 5:
    print("✅ LOO CV подтверждает 5-Fold CV — модель стабильна.")
elif abs(mae_loo_dur - 23.3) < 10:
    print("⚠️  LOO CV немного отличается от 5-Fold — модель в целом стабильна.")
else:
    print("❌ LOO CV значительно отличается — возможна нестабильность. Требуется анализ.")

print(f"\n5-Fold MAE: 23.3 дней")
print(f"LOO MAE:    {mae_loo_dur:.1f} дней")
print(f"Разница:    {mae_loo_dur - 23.3:+.1f} дней")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)