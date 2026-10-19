# 02_train_model.py
# Обучение Random Forest на группе А + перекалибровка по секторам

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "group_a.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

# === 2. ЗАГРУЗКА ===
df = pd.read_csv(INPUT_FILE)
print("=" * 60)
print("ЗАГРУЗКА ГРУППЫ А")
print("=" * 60)
print(f"Проектов: {len(df)}")
print(f"Колонок: {len(df.columns)}")

# === 3. ПЕРЕИМЕНОВАНИЕ КОЛОНОК ЧУВСТВИТЕЛЬНОСТИ ===
# Колонки 25-81 в исходном порядке = группы метрик с суффиксами
# Сопоставим их по индексам в df.columns

cols = list(df.columns)

# Названия групп метрик в порядке следования (из анализа DSLIB)
sensitivity_groups = [
    "CI", "SI", "SSI",
    "CRI_r", "CRI_rho", "CRI_tau",
    "CRI_r_cost", "CRI_rho_cost", "CRI_tau_cost",
    "CRI_r_res", "CRI_rho_res", "CRI_tau_res",
]

perf_groups = ["CV", "SV", "SV_t", "CPI", "SPI", "SPI_t", "p_factor"]

# Находим индекс первой колонки "avg" (обычно 25)
try:
    start_idx = cols.index("avg")
except ValueError:
    print("❌ Не найдена колонка 'avg'. Проверь структуру.")
    exit(1)

# Переименовываем первые 12 групп × 3 = 36 колонок (avg/std/skew)
rename_map = {}
idx = start_idx
for group in sensitivity_groups:
    for metric_suffix, metric_name in zip(
        ["avg", "std. dev.", "skew"],
        ["avg", "std", "skew"]
    ):
        if idx < len(cols):
            old = cols[idx]
            new = f"{group}_{metric_name}"
            rename_map[old] = new
            idx += 1

# Теперь idx указывает на начало performance metrics (там avg/std/final)
# Но там уже встречались "final" отдельно (в конце каждой группы)
# Пропустим "final" от последней sensitivity группы (она была skew)
# На самом деле для performance metrics формат: avg, std, final

# Проверим, что перед perf_groups стоит 'final' от последней sensitivity-группы
# Сдвигаем idx на 1, если там 'final' (это остаток CRI_r_res_skew? нет — там skew)
# По структуре: 12 групп × 3 = 36 колонок от start_idx
# Значит, idx = start_idx + 36 = начало performance metrics

perf_start_idx = start_idx + 36

# Переименовываем 7 групп × 3 = 21 колонку
idx = perf_start_idx
for group in perf_groups:
    for metric_suffix, metric_name in zip(
        ["avg", "std. dev.", "final"],
        ["avg", "std", "final"]
    ):
        if idx < len(cols):
            old = cols[idx]
            new = f"{group}_{metric_name}"
            # Не переименовываем целевые Duration и Cost
            if old in ["Duration", "Cost"]:
                break
            rename_map[old] = new
            idx += 1

df = df.rename(columns=rename_map)

# === 4. УНИФИКАЦИЯ SECTOR ===
df["Sector"] = df["Sector"].astype(str).str.lower().str.strip()

print("\n" + "=" * 60)
print("РАСПРЕДЕЛЕНИЕ ПО СЕКТОРАМ (после унификации)")
print("=" * 60)
print(df["Sector"].value_counts())

# === 5. ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ ===
base_features = [
    "# activities", "PD (days)", "BAC",
    "SP", "AD", "LA", "TF", "RI",
]

# Колонки чувствительности (после переименования)
sensitivity_features = [c for c in df.columns if any(
    c.startswith(g + "_") for g in sensitivity_groups + perf_groups
)]

# Sector в One-Hot
sector_dummies = pd.get_dummies(df["Sector"], prefix="sector")
df = pd.concat([df, sector_dummies], axis=1)
sector_features = list(sector_dummies.columns)

all_features = base_features + sensitivity_features + sector_features

# Проверяем, что все признаки есть в df
missing = [f for f in all_features if f not in df.columns]
if missing:
    print(f"\n⚠️  Отсутствуют признаки: {missing}")
    all_features = [f for f in all_features if f in df.columns]

print(f"\nПризнаков всего: {len(all_features)}")
print(f"  - Базовые: {len(base_features)}")
print(f"  - Чувствительность: {len(sensitivity_features)}")
print(f"  - Sector (One-Hot): {len(sector_features)}")

# === 6. АНАЛИЗ ПРОПУСКОВ ===
print("\n" + "=" * 60)
print("АНАЛИЗ ПРОПУСКОВ ПО ПРИЗНАКАМ")
print("=" * 60)

X = df[all_features].copy()
y = df["Duration"].copy()

# Приводим все признаки к числам
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

missing_pct = X.isna().mean() * 100
missing_table = pd.DataFrame({
    "Признак": missing_pct.index,
    "Пропусков, %": missing_pct.values.round(1)
}).sort_values("Пропусков, %", ascending=False)

print(missing_table.to_string(index=False))

# Удаляем признаки с > 50% пропусков
drop_cols = missing_pct[missing_pct > 50].index.tolist()
if drop_cols:
    print(f"\n⚠️  Удаляем признаки с > 50% пропусков ({len(drop_cols)}):")
    for c in drop_cols:
        print(f"   - {c}")
    X = X.drop(columns=drop_cols)
    all_features = [f for f in all_features if f not in drop_cols]

# Заполняем оставшиеся пропуски медианой
X = X.fillna(X.median(numeric_only=True))

print(f"\n✅ Итоговое число признаков: {X.shape[1]}")

# === 7. BASELINE ===
print("\n" + "=" * 60)
print("BASELINE: Duration = PD")
print("=" * 60)

pd_values = df["PD (days)"].astype(float)
valid_mask = pd_values.notna() & y.notna()

mae_base = mean_absolute_error(y[valid_mask], pd_values[valid_mask])
rmse_base = np.sqrt(mean_squared_error(y[valid_mask], pd_values[valid_mask]))
r2_base = r2_score(y[valid_mask], pd_values[valid_mask])

# MAPE по отклонению
deviation_real = (y[valid_mask] - pd_values[valid_mask]) / pd_values[valid_mask]
mape_base = np.mean(np.abs(deviation_real)) * 100

print(f"MAE:  {mae_base:.1f} дней")
print(f"RMSE: {rmse_base:.1f} дней")
print(f"MAPE (по отклонению): {mape_base:.1f}%")
print(f"R²:   {r2_base:.3f}")

# === 8. КРОСС-ВАЛИДАЦИЯ ===
print("\n" + "=" * 60)
print("5-FOLD КРОСС-ВАЛИДАЦИЯ")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Массивы для предсказаний
oof_pred = np.zeros(len(X))         # out-of-fold предсказания глобальной модели
oof_pred_cal = np.zeros(len(X))     # после перекалибровки

# Словарь для перекалибровки по секторам
sectors = df["Sector"].values

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

    pred_val = model.predict(X_val)
    oof_pred[val_idx] = pred_val

    # Перекалибровка по секторам
    # Считаем остатки на train
    pred_train = model.predict(X_train)
    residuals_train = y_train.values - pred_train

    # Для каждого сектора в train — медиана остатка
    train_sectors = sectors[train_idx]
    sector_correction = {}
    for s in np.unique(train_sectors):
        mask = train_sectors == s
        n = mask.sum()
        if n >= 10:  # корректируем только крупные секторы
            sector_correction[s] = np.median(residuals_train[mask])
        else:
            sector_correction[s] = 0.0

    # Применяем корректировку к val
    val_sectors = sectors[val_idx]
    correction_val = np.array([sector_correction.get(s, 0.0) for s in val_sectors])
    oof_pred_cal[val_idx] = pred_val + correction_val

    print(f"Fold {fold}: MAE={mean_absolute_error(y_val, pred_val):.1f} "
          f"→ после корр. {mean_absolute_error(y_val, oof_pred_cal[val_idx]):.1f}")

# === 9. МЕТРИКИ ПОСЛЕ КРОСС-ВАЛИДАЦИИ ===
print("\n" + "=" * 60)
print("ИТОГОВЫЕ МЕТРИКИ (5-FOLD OOF)")
print("=" * 60)

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dev = (y_true - y_pred) / y_true
    mape = np.mean(np.abs(dev)) * 100
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.1f} дней")
    print(f"  RMSE: {rmse:.1f} дней")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  R²:   {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

metrics_base = evaluate(y, pd_values.fillna(pd_values.median()), "BASELINE (Duration = PD)")
metrics_global = evaluate(y, oof_pred, "ГЛОБАЛЬНАЯ RF")
metrics_cal = evaluate(y, oof_pred_cal, "RF + ПЕРЕКАЛИБРОВКА ПО СЕКТОРАМ")

# === 10. МЕТРИКИ ПО СЕКТОРАМ ===
print("\n" + "=" * 60)
print("МЕТРИКИ ПО СЕКТОРАМ (RF + перекалибровка)")
print("=" * 60)

for s in sorted(np.unique(sectors)):
    mask = sectors == s
    n = mask.sum()
    if n < 3:
        continue
    y_s = y[mask]
    p_s = oof_pred_cal[mask]
    mae_s = mean_absolute_error(y_s, p_s)
    mape_s = np.mean(np.abs((y_s - p_s) / y_s)) * 100
    print(f"\n{s} (n={n}):")
    print(f"  MAE: {mae_s:.1f} дней, MAPE: {mape_s:.1f}%")

# === 11. ОБУЧАЕМ ФИНАЛЬНУЮ МОДЕЛЬ НА ВСЕЙ ВЫБОРКЕ ===
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА ВСЕЙ ВЫБОРКЕ")
print("=" * 60)

final_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y)

# Считаем перекалибровку по секторам на всей выборке
pred_all = final_model.predict(X)
residuals_all = y.values - pred_all
sector_correction_final = {}
for s in np.unique(sectors):
    mask = sectors == s
    n = mask.sum()
    if n >= 10:
        sector_correction_final[s] = float(np.median(residuals_all[mask]))
    else:
        sector_correction_final[s] = 0.0

print("Корректировки по секторам (дней):")
for s, c in sorted(sector_correction_final.items()):
    n = (sectors == s).sum()
    print(f"  {s} (n={n}): {c:+.1f}")

# Сохраняем модель и коррекции
joblib.dump({
    "model": final_model,
    "features": all_features,
    "sector_correction": sector_correction_final,
}, MODEL_DIR / "rf_model.pkl")

print(f"\n✅ Модель сохранена: {MODEL_DIR / 'rf_model.pkl'}")

# === 12. ВАЖНОСТЬ ПРИЗНАКОВ ===
print("\n" + "=" * 60)
print("ТОП-20 ВАЖНЫХ ПРИЗНАКОВ")
print("=" * 60)

importances = pd.Series(final_model.feature_importances_, index=all_features)
importances = importances.sort_values(ascending=False)
print(importances.head(20).to_string())

# === 13. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
results = pd.DataFrame({
    "Code": df["Code"],
    "Sector": df["Sector"],
    "Duration_real": y,
    "Duration_baseline_PD": pd_values,
    "Duration_pred_global": oof_pred,
    "Duration_pred_calibrated": oof_pred_cal,
})
results.to_csv(OUTPUT_DIR / "predictions_group_a.csv", index=False)
print(f"\n✅ Предсказания сохранены: {OUTPUT_DIR / 'predictions_group_a.csv'}")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)