# 03_3_winsorization.py
# Защита от выбросов: winsorization + переобучение QRF

import pandas as pd
import numpy as np
from pathlib import Path
from quantile_forest import RandomForestQuantileRegressor
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === 1. ПУТИ ===
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "group_a.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

# === 2. ЗАГРУЗКА ===
df = pd.read_csv(INPUT_FILE)
print("=" * 60)
print("ЗАЩИТА ОТ ВЫБРОСОВ (WINSORIZATION)")
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
    for _, metric_name in zip(["avg", "std. dev.", "skew"], ["avg", "std", "skew"]):
        if idx < len(cols):
            rename_map[cols[idx]] = f"{group}_{metric_name}"
            idx += 1
perf_start_idx = start_idx + 36
idx = perf_start_idx
for group in perf_groups:
    for _, metric_name in zip(["avg", "std. dev.", "final"], ["avg", "std", "final"]):
        if idx < len(cols):
            old = cols[idx]
            if old in ["Duration", "Cost"]:
                break
            rename_map[old] = f"{group}_{metric_name}"
            idx += 1

df = df.rename(columns=rename_map)
df["Sector"] = df["Sector"].astype(str).str.lower().str.strip()
sector_dummies = pd.get_dummies(df["Sector"], prefix="sector")
df = pd.concat([df, sector_dummies], axis=1)

df["PD_num"] = pd.to_numeric(df["PD (days)"], errors="coerce")
df["Duration_num"] = pd.to_numeric(df["Duration"], errors="coerce")
df["Early_late_calc"] = (df["Duration_num"] - df["PD_num"]) / df["PD_num"]

valid_mask = df["Early_late_calc"].notna() & df["PD_num"].notna()
df = df[valid_mask].reset_index(drop=True)

y_dev = df["Early_late_calc"].copy()
PD = df["PD_num"].values

# === 4. АНАЛИЗ ВЫБРОСОВ ===
print("\n" + "=" * 60)
print("АНАЛИЗ ВЫБРОСОВ")
print("=" * 60)

print(f"\nРаспределение Early/late:")
print(f"  Среднее:  {y_dev.mean():+.3f}")
print(f"  Медиана:  {y_dev.median():+.3f}")
print(f"  Std:      {y_dev.std():.3f}")
print(f"  Min:      {y_dev.min():+.3f}")
print(f"  P01:      {y_dev.quantile(0.01):+.3f}")
print(f"  P05:      {y_dev.quantile(0.05):+.3f}")
print(f"  P95:      {y_dev.quantile(0.95):+.3f}")
print(f"  P99:      {y_dev.quantile(0.99):+.3f}")
print(f"  Max:      {y_dev.max():+.3f}")

# Считаем выбросы по разным порогам
print(f"\nВыбросы по порогам:")
for threshold in [0.5, 0.8, 1.0]:
    n_below = (y_dev < -threshold).sum()
    n_above = (y_dev > threshold).sum()
    print(f"  |Early/late| > {threshold}: {n_below + n_above} проектов "
          f"(низ: {n_below}, верх: {n_above})")

# Показываем экстремальные проекты
print(f"\nПроекты с |Early/late| > 0.5:")
extreme = df[df["Early_late_calc"].abs() > 0.5][
    ["Code", "Sector", "PD_num", "Duration_num", "Early_late_calc"]
].copy()
extreme["Early_late_calc"] = extreme["Early_late_calc"].round(3)
extreme = extreme.sort_values("Early_late_calc")
print(extreme.to_string(index=False))

# === 5. WINSORIZATION ===
print("\n" + "=" * 60)
print("WINSORIZATION [−0.5, +1.0]")
print("=" * 60)

LOWER = -0.5
UPPER = 1.0

y_dev_winsor = y_dev.clip(lower=LOWER, upper=UPPER)

n_changed = (y_dev != y_dev_winsor).sum()
print(f"\nИзменено проектов: {n_changed}")
print(f"Было:  min={y_dev.min():+.3f}, max={y_dev.max():+.3f}")
print(f"Стало: min={y_dev_winsor.min():+.3f}, max={y_dev_winsor.max():+.3f}")

# === 6. ПОДГОТОВКА X ===
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

missing_pct = X.isna().mean() * 100
drop_cols = missing_pct[missing_pct > 50].index.tolist()
if drop_cols:
    X = X.drop(columns=drop_cols)
    all_features = [f for f in all_features if f not in drop_cols]

X = X.fillna(X.median(numeric_only=True))
print(f"\nПризнаков: {X.shape[1]}")

# === 7. ОБУЧЕНИЕ QRF НА ОГРАНИЧЕННЫХ ДАННЫХ ===
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ QRF С ЗАЩИТОЙ")
print("=" * 60)

qrf = RandomForestQuantileRegressor(
    n_estimators=500,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
qrf.fit(X, y_dev_winsor)
print(f"✅ Модель обучена")

# === 8. СРАВНЕНИЕ КВАНТИЛЕЙ ===
print("\n" + "=" * 60)
print("СРАВНЕНИЕ КВАНТИЛЕЙ (до и после)")
print("=" * 60)

quantiles = [0.10, 0.25, 0.50, 0.75, 0.80, 0.90]
pred_q = qrf.predict(X, quantiles=quantiles)

# Ограничиваем предсказания тоже
pred_q_clipped = np.clip(pred_q, LOWER, UPPER)

print(f"\n{'Квантиль':<10} {'Среднее':>10} {'P10':>10} {'P90':>10}")
print(f"{'-' * 45}")
for i, q in enumerate(quantiles):
    vals = pred_q[:, i]
    clipped = pred_q_clipped[:, i]
    print(f"P{int(q*100):<9} {vals.mean():>+10.3f} {vals.min():>+10.3f} {vals.max():>+10.3f}")

# === 9. ГЛАВНАЯ ПРОВЕРКА: P10 для проблемного проекта ===
print("\n" + "=" * 60)
print("ПРОВЕРКА: P10 ДЛЯ ПРОБЛЕМНЫХ ПРОЕКТОВ")
print("=" * 60)

# Ищем проект с наибольшим фактическим отклонением
worst_idx = y_dev.idxmax()
worst_code = df.loc[worst_idx, "Code"]
worst_real = y_dev.loc[worst_idx]

print(f"\nСамый проблемный проект: {worst_code}")
print(f"  Фактическое отклонение: {worst_real:+.3f}")
print(f"  P10 предсказания (новое): {pred_q_clipped[worst_idx, 0]:+.3f}")
print(f"  P50 предсказания (новое): {pred_q_clipped[worst_idx, 2]:+.3f}")
print(f"  P90 предсказания (новое): {pred_q_clipped[worst_idx, 5]:+.3f}")

# Аналогично для самого «быстрого» проекта
best_idx = y_dev.idxmin()
best_code = df.loc[best_idx, "Code"]
best_real = y_dev.loc[best_idx]

print(f"\nСамый быстрый проект: {best_code}")
print(f"  Фактическое отклонение: {best_real:+.3f}")
print(f"  P10 предсказания (новое): {pred_q_clipped[best_idx, 0]:+.3f}")
print(f"  P50 предсказания (новое): {pred_q_clipped[best_idx, 2]:+.3f}")

# === 10. ПЕРЕСЧЁТ ВЕРОЯТНОСТЕЙ ===
print("\n" + "=" * 60)
print("СРЕДНИЕ ВЕРОЯТНОСТИ (ПОСЛЕ ЗАЩИТЫ)")
print("=" * 60)

# Считаем P(Duration <= PD) для каждого проекта
all_qs = [round(q, 2) for q in np.arange(0.01, 1.0, 0.01)]
pred_all_q = qrf.predict(X, quantiles=all_qs)
pred_all_q_clipped = np.clip(pred_all_q, LOWER, UPPER)

# Для каждого проекта считаем вероятность Early_late <= 0
probs = []
for i in range(len(X)):
    p = np.mean(pred_all_q_clipped[i] <= 0)
    probs.append(p)

print(f"\n  P(уложиться в план): {np.mean(probs)*100:.1f}%")
print(f"  Было (без защиты):   29.7%")

# === 11. СОХРАНЕНИЕ МОДЕЛИ ===
joblib.dump({
    "model": qrf,
    "features": all_features,
    "quantiles": quantiles,
    "winsor_bounds": (LOWER, UPPER),
}, MODEL_DIR / "qrf_model_winsor.pkl")
print(f"\n✅ Модель: {MODEL_DIR / 'qrf_model_winsor.pkl'}")

# === 12. СОХРАНЕНИЕ ПРЕДСКАЗАНИЙ ===
results = pd.DataFrame({
    "Code": df["Code"],
    "Sector": df["Sector"],
    "PD": PD,
    "Duration_real": df["Duration_num"],
    "Early_late_real": y_dev,
    "Early_late_winsor": y_dev_winsor,
    "P10_dev": pred_q_clipped[:, 0],
    "P50_dev": pred_q_clipped[:, 2],
    "P80_dev": pred_q_clipped[:, 4],
    "P90_dev": pred_q_clipped[:, 5],
    "P10_dur": PD * (1 + pred_q_clipped[:, 0]),
    "P50_dur": PD * (1 + pred_q_clipped[:, 2]),
    "P80_dur": PD * (1 + pred_q_clipped[:, 4]),
    "P90_dur": PD * (1 + pred_q_clipped[:, 5]),
})
results.to_csv(OUTPUT_DIR / "predictions_winsor.csv", index=False)
print(f"✅ Предсказания: {OUTPUT_DIR / 'predictions_winsor.csv'}")

# === 13. ГРАФИК: СРАВНЕНИЕ P10 ДО/ПОСЛЕ ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Загружаем старую модель для сравнения
old_qrf_data = joblib.load(MODEL_DIR / "qrf_model.pkl")
old_qrf = old_qrf_data["model"]
old_pred = old_qrf.predict(X, quantiles=quantiles)

# График 1: распределение P10
ax = axes[0]
ax.hist(old_pred[:, 0] * 100, bins=30, alpha=0.5, color="red", label="Без защиты")
ax.hist(pred_q_clipped[:, 0] * 100, bins=30, alpha=0.5, color="green", label="С защитой")
ax.axvline(-50, color="black", linestyle="--", label="Граница −50%")
ax.set_xlabel("P10 отклонения (%)")
ax.set_ylabel("Количество проектов")
ax.set_title("P10: до и после winsorization")
ax.legend()
ax.grid(alpha=0.3)

# График 2: распределение P90
ax = axes[1]
ax.hist(old_pred[:, 5] * 100, bins=30, alpha=0.5, color="red", label="Без защиты")
ax.hist(pred_q_clipped[:, 5] * 100, bins=30, alpha=0.5, color="green", label="С защитой")
ax.axvline(100, color="black", linestyle="--", label="Граница +100%")
ax.set_xlabel("P90 отклонения (%)")
ax.set_ylabel("Количество проектов")
ax.set_title("P90: до и после winsorization")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "winsor_comparison.png", dpi=150)
plt.close()
print(f"✅ График: {OUTPUT_DIR / 'winsor_comparison.png'}")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)