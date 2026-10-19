# 03_1_scenario_analysis.py
# Анализ сценариев: P(Duration <= T) для разных целевых сроков T

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

# === 2. ЗАГРУЗКА МОДЕЛИ ===
qrf_data = joblib.load(MODEL_FILE)
qrf = qrf_data["model"]
all_features = qrf_data["features"]

print("=" * 60)
print("АНАЛИЗ СЦЕНАРИЕВ: P(Duration <= T)")
print("=" * 60)

# === 3. ЗАГРУЗКА ДАННЫХ ===
df = pd.read_csv(INPUT_FILE)

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

# Медианы
X_all = df[all_features].copy()
for col in X_all.columns:
    X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
X_all = X_all.fillna(X_all.median(numeric_only=True))
medians = X_all.median(numeric_only=True).to_dict()

# === 4. ФУНКЦИЯ: P(Duration <= T) ===
def prob_leq_T(PD, T, features_dict, qrf, all_features, medians):
    """
    P(Duration <= T) для проекта с признаками features_dict.
    Эквивалентно: P(Early_late <= T/PD - 1).
    """
    row = {f: features_dict.get(f, 0) for f in all_features}
    row_df = pd.DataFrame([row])
    for f in all_features:
        if pd.isna(row_df[f].iloc[0]):
            row_df[f] = medians.get(f, 0)

    # Предсказанные квантили отклонения
    qs = [round(q, 2) for q in np.arange(0.01, 1.0, 0.01)]
    pred_q = qrf.predict(row_df, quantiles=qs)[0]

    # Условие: PD * (1 + Early_late) <= T
    # => Early_late <= T/PD - 1
    threshold = T / PD - 1.0
    return float(np.mean(pred_q <= threshold))

# === 5. ВЫБИРАЕМ 3 ПРОЕКТА ===
df_valid = df[df["PD_num"].notna()].copy().sort_values("PD_num").reset_index(drop=True)

projects = [
    ("Маленький проект", df_valid.iloc[len(df_valid) // 4]),
    ("Средний проект", df_valid.iloc[len(df_valid) // 2]),
    ("Большой проект", df_valid.iloc[(3 * len(df_valid)) // 4]),
]

# === 6. ТАБЛИЦА СЦЕНАРИЕВ ===
print("\n" + "=" * 60)
print("ТАБЛИЦА: P(Duration <= T) для разных T")
print("=" * 60)

all_results = []

for label, project in projects:
    code = project["Code"]
    pd_original = project["PD_num"]
    duration_real = project["Duration_num"]
    sector = project["Sector"]

    print(f"\n{'─' * 60}")
    print(f"{label}: {code} ({sector})")
    print(f"  План (PD): {pd_original:.0f} дней")
    if pd.notna(duration_real):
        print(f"  Факт:       {duration_real:.0f} дней "
              f"({(duration_real - pd_original) / pd_original * 100:+.1f}%)")
    print(f"{'─' * 60}")

    features = project[all_features].to_dict()
    for f in all_features:
        if pd.isna(features.get(f)):
            features[f] = medians.get(f, 0)

    # Множители для целевого срока T
    multipliers = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    print(f"\n{'T (дни)':>10} {'T/PD':>8} {'P(Dur ≤ T)':>12}")
    print(f"{'─' * 60}")

    for mult in multipliers:
        T = pd_original * mult
        p = prob_leq_T(pd_original, T, features, qrf, all_features, medians)
        print(f"{T:>10.0f} {mult:>8.2f} {p*100:>11.1f}%")

        all_results.append({
            "project": code,
            "label": label,
            "PD": pd_original,
            "T": T,
            "multiplier": mult,
            "P_dur_leq_T": p,
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "scenario_analysis.csv", index=False)

# === 7. ГРАФИК: P(Duration <= T) vs T ===
fig, ax = plt.subplots(figsize=(12, 7))

for label, project in projects:
    code = project["Code"]
    pd_original = project["PD_num"]

    # Широкий диапазон T
    T_range = np.linspace(pd_original * 0.9, pd_original * 2.0, 50)
    probs = []

    features = project[all_features].to_dict()
    for f in all_features:
        if pd.isna(features.get(f)):
            features[f] = medians.get(f, 0)

    for T in T_range:
        probs.append(prob_leq_T(pd_original, T, features, qrf, all_features, medians))

    ax.plot(T_range, np.array(probs) * 100, marker="o", markersize=3,
            label=f"{label} ({code}, PD={pd_original:.0f})")

    # Отметим PD на графике
    ax.axvline(pd_original, color="gray", linestyle=":", alpha=0.3)

ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% порог")
ax.axhline(80, color="gray", linestyle=":", alpha=0.5, label="80% порог")
ax.set_xlabel("Целевой срок T (дни)", fontsize=12)
ax.set_ylabel("P(Duration ≤ T), %", fontsize=12)
ax.set_title("Вероятность завершить проект в срок T", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scenario_curve.png", dpi=150)
plt.close()
print(f"\n✅ График: {OUTPUT_DIR / 'scenario_curve.png'}")
print(f"✅ Таблица: {OUTPUT_DIR / 'scenario_analysis.csv'}")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)