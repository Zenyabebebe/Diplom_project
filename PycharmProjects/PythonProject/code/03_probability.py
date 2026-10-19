# 03_probability.py
# Вероятностный прогноз: Quantile Regression Forest
# Даёт P10, P50, P90 отклонения и вероятности попадания в интервалы

import pandas as pd
import numpy as np
from pathlib import Path
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import KFold
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
print("ВЕРОЯТНОСТНЫЙ ПРОГНОЗ (Quantile Regression Forest)")
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
    for _, metric_name in zip(
        ["avg", "std. dev.", "skew"], ["avg", "std", "skew"]
    ):
        if idx < len(cols):
            rename_map[cols[idx]] = f"{group}_{metric_name}"
            idx += 1

perf_start_idx = start_idx + 36
idx = perf_start_idx
for group in perf_groups:
    for _, metric_name in zip(
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

# === 7. QUANTILE REGRESSION FOREST ===
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ QUANTILE REGRESSION FOREST")
print("=" * 60)

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

qrf = RandomForestQuantileRegressor(
    n_estimators=500,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
qrf.fit(X, y_dev)

print(f"✅ Модель обучена на {len(X)} проектах")
print(f"Квантили: {quantiles}")

# === 8. ПРЕДСКАЗАНИЯ НА ВСЕЙ ВЫБОРКЕ ===
pred_quantiles = qrf.predict(X, quantiles=quantiles)

# pred_quantiles — массив [n_samples, n_quantiles]
# Переводим в Duration
pd_arr = PD.reshape(-1, 1)
dur_quantiles = pd_arr * (1 + pred_quantiles)

print("\n" + "=" * 60)
print("СРЕДНИЕ КВАНТИЛИ ПО ГРУППЕ А")
print("=" * 60)
for i, q in enumerate(quantiles):
    mean_dev = pred_quantiles[:, i].mean()
    mean_dur = dur_quantiles[:, i].mean()
    print(f"  P{int(q*100):>2}: отклонение {mean_dev:+.3f}, срок {mean_dur:.1f} дней")

# === 9. ВЕРОЯТНОСТИ ПОПАДАНИЯ В ИНТЕРВАЛЫ ===
# Для каждого проекта считаем P(Duration <= threshold)
# через эмпирическое распределение деревьев

print("\n" + "=" * 60)
print("ВЕРОЯТНОСТИ ПОПАДАНИЯ В ИНТЕРВАЛЫ")
print("=" * 60)

# Квантили для расчёта вероятностей
all_quantiles = [round(q, 2) for q in np.arange(0.01, 1.0, 0.01)]
pred_all_q = qrf.predict(X, quantiles=all_quantiles)
dur_all_q = pd_arr * (1 + pred_all_q)  # [n_samples, n_quantiles]

def prob_leq(dur_quantiles_row, threshold):
    """P(Duration <= threshold) по квантилям"""
    return np.mean(dur_quantiles_row <= threshold)

# Считаем для каждого проекта
results = []
for i in range(len(X)):
    pd_i = PD[i]
    dur_row = dur_all_q[i]
    results.append({
        "Code": df["Code"].iloc[i],
        "Sector": df["Sector"].iloc[i],
        "PD": pd_i,
        "Duration_real": Duration_real[i],
        "P10_dur": np.quantile(dur_row, 0.10),
        "P50_dur": np.quantile(dur_row, 0.50),
        "P90_dur": np.quantile(dur_row, 0.90),
        "P_leq_PD": prob_leq(dur_row, pd_i),
        "P_leq_PD_105": prob_leq(dur_row, pd_i * 1.05),
        "P_leq_PD_110": prob_leq(dur_row, pd_i * 1.10),
        "P_leq_PD_120": prob_leq(dur_row, pd_i * 1.20),
        "P_leq_PD_130": prob_leq(dur_row, pd_i * 1.30),
    })

results_df = pd.DataFrame(results)

# Средние вероятности
print("\nСредние вероятности по группе А:")
print(f"  P(уложиться в план):       {results_df['P_leq_PD'].mean()*100:.1f}%")
print(f"  P(задержка до +5%):        {results_df['P_leq_PD_105'].mean()*100:.1f}%")
print(f"  P(задержка до +10%):       {results_df['P_leq_PD_110'].mean()*100:.1f}%")
print(f"  P(задержка до +20%):       {results_df['P_leq_PD_120'].mean()*100:.1f}%")
print(f"  P(задержка до +30%):       {results_df['P_leq_PD_130'].mean()*100:.1f}%")

# === 10. СОХРАНЕНИЕ ===
results_df.to_csv(OUTPUT_DIR / "probability_predictions.csv", index=False)
print(f"\n✅ Вероятности сохранены: {OUTPUT_DIR / 'probability_predictions.csv'}")

# Сохраняем модель
joblib.dump({
    "model": qrf,
    "features": all_features,
    "quantiles": quantiles,
}, MODEL_DIR / "qrf_model.pkl")
print(f"✅ Модель сохранена: {MODEL_DIR / 'qrf_model.pkl'}")

# === 11. ПРИМЕРЫ ПРОГНОЗОВ ===
print("\n" + "=" * 60)
print("ПРИМЕРЫ ПРОГНОЗОВ (первые 10 проектов)")
print("=" * 60)

example_cols = ["Code", "PD", "Duration_real", "P10_dur", "P50_dur", "P90_dur", "P_leq_PD"]
example = results_df[example_cols].head(10).copy()
for col in ["P10_dur", "P50_dur", "P90_dur"]:
    example[col] = example[col].round(0).astype(int)
example["P_leq_PD"] = (example["P_leq_PD"] * 100).round(0).astype(int)

print(example.to_string(index=False))

# === 12. ГРАФИК ДЛЯ ОДНОГО ПРИМЕРА ===
print("\n" + "=" * 60)
print("ГРАФИК РАСПРЕДЕЛЕНИЯ (проект с PD = 300)")
print("=" * 60)

# Найдём проект с PD около 300
idx_example = (PD > 250).argmax() if (PD > 250).any() else 0
pd_example = PD[idx_example]
dur_row = dur_all_q[idx_example]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(dur_row, bins=40, alpha=0.7, color="steelblue", edgecolor="black")
ax.axvline(pd_example, color="red", linestyle="--", linewidth=2, label=f"PD = {pd_example:.0f}")
ax.axvline(np.quantile(dur_row, 0.5), color="green", linestyle="-", linewidth=2, label=f"P50 = {np.quantile(dur_row, 0.5):.0f}")
ax.axvline(np.quantile(dur_row, 0.1), color="orange", linestyle=":", linewidth=2, label=f"P10 = {np.quantile(dur_row, 0.1):.0f}")
ax.axvline(np.quantile(dur_row, 0.9), color="orange", linestyle=":", linewidth=2, label=f"P90 = {np.quantile(dur_row, 0.9):.0f}")
ax.set_xlabel("Срок (дни)")
ax.set_ylabel("Частота")
ax.set_title(f"Распределение прогноза срока\nПроект {df['Code'].iloc[idx_example]} (PD = {pd_example:.0f} дней)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "probability_distribution_example.png", dpi=150)
plt.close()
print(f"✅ График: {OUTPUT_DIR / 'probability_distribution_example.png'}")

# === 13. ФУНКЦИЯ ИНТЕРАКТИВНОГО ПРОГНОЗА ===
print("\n" + "=" * 60)
print("ФУНКЦИЯ ИНТЕРАКТИВНОГО ПРОГНОЗА")
print("=" * 60)

def predict_project(PD, features_dict):
    """
    Прогноз срока и вероятностей для одного проекта.

    Параметры:
    - PD: плановый срок (дни)
    - features_dict: словарь признаков (ключ: значение)

    Возвращает:
    - dict с полями: P10, P50, P90, вероятности интервалов
    """
    # Создаём DataFrame из одной строки
    row = {f: features_dict.get(f, 0) for f in all_features}

    # Медианы для заполнения пропусков
    medians = X.median(numeric_only=True)
    row_df = pd.DataFrame([row])
    for f in all_features:
        if pd.isna(row_df[f].iloc[0]):
            row_df[f] = medians[f]

    # Предсказание квантилей
    qs = [round(q, 2) for q in np.arange(0.01, 1.0, 0.01)]
    pred_q = qrf.predict(row_df, quantiles=qs)[0]
    dur_q = PD * (1 + pred_q)

    return {
        "P10": float(np.quantile(dur_q, 0.10)),
        "P50": float(np.quantile(dur_q, 0.50)),
        "P90": float(np.quantile(dur_q, 0.90)),
        "P(<=PD)": float(np.mean(dur_q <= PD)),
        "P(<=PD*1.05)": float(np.mean(dur_q <= PD * 1.05)),
        "P(<=PD*1.10)": float(np.mean(dur_q <= PD * 1.10)),
        "P(<=PD*1.20)": float(np.mean(dur_q <= PD * 1.20)),
        "P(<=PD*1.30)": float(np.mean(dur_q <= PD * 1.30)),
    }

# Пример использования
print("\nПример: PD = 300 дней, средние признаки проекта")
print("(признаки берём средние по выборке — для демонстрации)")

example_features = X.mean().to_dict()
example_result = predict_project(300, example_features)
print("\nПрогноз для PD = 300 дней:")
for k, v in example_result.items():
    if "P(" in k:
        print(f"  {k}: {v*100:.1f}%")
    else:
        print(f"  {k}: {v:.0f} дней")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)