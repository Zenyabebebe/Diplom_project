# 03_2_buffer_analysis.py
# Сводная таблица буферов: сколько нужно запаса для P50, P80, P90

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# === 1. ПУТИ ===
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "group_a.csv"
MODEL_FILE = BASE_DIR / "models" / "qrf_model.pkl"
OUTPUT_DIR = BASE_DIR / "output"

# === 2. ЗАГРУЗКА ===
qrf_data = joblib.load(MODEL_FILE)
qrf = qrf_data["model"]
all_features = qrf_data["features"]

print("=" * 60)
print("СВОДНАЯ ТАБЛИЦА БУФЕРОВ ВРЕМЕНИ")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

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

# Готовим X
X = df[all_features].copy()
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.fillna(X.median(numeric_only=True))

PD = df["PD_num"].values

# === 4. ПРЕДСКАЗАНИЕ КВАНТИЛЕЙ ДЛЯ ВСЕХ ПРОЕКТОВ ===
print("\nСчитаем квантили для всех проектов...")

qs = [0.10, 0.25, 0.50, 0.75, 0.80, 0.90]
pred_q = qrf.predict(X, quantiles=qs)

# Переводим в срок
dur_q = PD.reshape(-1, 1) * (1 + pred_q)

# Считаем буферы (% отклонения от PD)
buf_q = (dur_q - PD.reshape(-1, 1)) / PD.reshape(-1, 1) * 100

# === 5. СВОДНАЯ ТАБЛИЦА ПО ВСЕМ ПРОЕКТАМ ===
summary = pd.DataFrame({
    "Code": df["Code"],
    "Sector": df["Sector"],
    "PD": PD,
    "Duration_real": df["Duration_num"].values,
    "Early_late_real": df["Early_late_calc"].values,
    "P10_dur": dur_q[:, 0],
    "P25_dur": dur_q[:, 1],
    "P50_dur": dur_q[:, 2],
    "P75_dur": dur_q[:, 3],
    "P80_dur": dur_q[:, 4],
    "P90_dur": dur_q[:, 5],
    "P10_buf": buf_q[:, 0],
    "P50_buf": buf_q[:, 2],
    "P80_buf": buf_q[:, 4],
    "P90_buf": buf_q[:, 5],
})

summary.to_csv(OUTPUT_DIR / "buffer_analysis.csv", index=False)
print(f"✅ Таблица: {OUTPUT_DIR / 'buffer_analysis.csv'}")

# === 6. СРЕДНИЕ БУФЕРЫ ПО ГРУППЕ ===
print("\n" + "=" * 60)
print("СРЕДНИЕ БУФЕРЫ ПО ГРУППЕ А (n=116)")
print("=" * 60)

print(f"\n{'Уровень':<10} {'Средний':>12} {'Медиана':>12} {'Std':>10}")
print(f"{'-' * 50}")
for name, buf_col in [
    ("P50", "P50_buf"),
    ("P80", "P80_buf"),
    ("P90", "P90_buf"),
]:
    mean_buf = summary[buf_col].mean()
    median_buf = summary[buf_col].median()
    std_buf = summary[buf_col].std()
    print(f"{name:<10} {mean_buf:>+11.1f}% {median_buf:>+11.1f}% {std_buf:>9.1f}%")

# === 7. РАЗБИВКА ПО СЕКТОРАМ ===
print("\n" + "=" * 60)
print("СРЕДНИЙ БУФЕР P80 ПО СЕКТОРАМ")
print("=" * 60)

for sector in sorted(df["Sector"].unique()):
    mask = df["Sector"] == sector
    n = mask.sum()
    mean_buf = summary.loc[mask, "P80_buf"].mean()
    median_buf = summary.loc[mask, "P80_buf"].median()
    print(f"\n{sector} (n={n}):")
    print(f"  Средний: {mean_buf:+.1f}%")
    print(f"  Медиана: {median_buf:+.1f}%")

# === 8. БУФЕР P80 vs PD ===
print("\n" + "=" * 60)
print("БУФЕР P80 vs РАЗМЕР ПРОЕКТА")
print("=" * 60)

bins = [0, 100, 200, 400, 10000]
labels = ["<100 дней", "100-200", "200-400", ">400"]
summary["size_cat"] = pd.cut(summary["PD"], bins=bins, labels=labels)

for cat in labels:
    subset = summary[summary["size_cat"] == cat]
    if len(subset) == 0:
        continue
    print(f"\n{cat} (n={len(subset)}):")
    print(f"  Средний PD:      {subset['PD'].mean():.0f} дней")
    print(f"  Буфер P50:       {subset['P50_buf'].mean():+.1f}%")
    print(f"  Буфер P80:       {subset['P80_buf'].mean():+.1f}%")
    print(f"  Буфер P90:       {subset['P90_buf'].mean():+.1f}%")

# === 9. ГРАФИК: БУФЕР P80 vs PD ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График 1: буфер P80 vs PD
ax = axes[0]
ax.scatter(summary["PD"], summary["P80_buf"], alpha=0.6, edgecolors="k", linewidths=0.5, color="steelblue")
ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Нет буфера")
ax.axhline(summary["P80_buf"].mean(), color="green", linestyle=":",
           label=f"Средний буфер = {summary['P80_buf'].mean():+.1f}%")
ax.set_xlabel("Плановый срок PD (дни)")
ax.set_ylabel("Буфер P80 (%)")
ax.set_title("Необходимый буфер для 80% уверенности")
ax.legend()
ax.grid(alpha=0.3)

# График 2: распределение буфера P80
ax = axes[1]
ax.hist(summary["P80_buf"], bins=25, color="steelblue", edgecolor="black", alpha=0.7)
ax.axvline(summary["P80_buf"].mean(), color="red", linestyle="--",
           label=f"Среднее = {summary['P80_buf'].mean():+.1f}%")
ax.axvline(summary["P80_buf"].median(), color="green", linestyle=":",
           label=f"Медиана = {summary['P80_buf'].median():+.1f}%")
ax.set_xlabel("Буфер P80 (%)")
ax.set_ylabel("Количество проектов")
ax.set_title("Распределение необходимого буфера")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "buffer_analysis.png", dpi=150)
plt.close()
print(f"\n✅ График: {OUTPUT_DIR / 'buffer_analysis.png'}")

# === 10. ПРИМЕРЫ ПРОЕКТОВ ===
print("\n" + "=" * 60)
print("ПРИМЕРЫ: 10 ПРОЕКТОВ")
print("=" * 60)

example = summary.head(10)[["Code", "Sector", "PD", "P50_dur", "P80_dur", "P90_dur", "P80_buf"]].copy()
for col in ["P50_dur", "P80_dur", "P90_dur"]:
    example[col] = example[col].round(0).astype(int)
example["P80_buf"] = example["P80_buf"].round(1)

print(example.to_string(index=False))

# === 11. ВЫВОДЫ ===
print("\n" + "=" * 60)
print("ВЫВОДЫ")
print("=" * 60)

mean_p80 = summary["P80_buf"].mean()

if mean_p80 < 20:
    verdict = "✅ Модель НЕ перестраховывается. Буфер разумный."
elif mean_p80 < 35:
    verdict = "✅ Буфер умеренный. Соответствует мировой практике."
elif mean_p80 < 50:
    verdict = "⚠️ Буфер высокий. Возможно, есть перестраховка."
else:
    verdict = "❌ Буфер очень высокий. Модель перестраховывается."

print(f"\nСредний буфер P80: {mean_p80:+.1f}%")
print(verdict)
print(f"\nДля сравнения: среднее фактическое отклонение в данных = "
      f"{df['Early_late_calc'].mean()*100:+.1f}%")
print(f"Медиана фактического отклонения = "
      f"{df['Early_late_calc'].median()*100:+.1f}%")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)