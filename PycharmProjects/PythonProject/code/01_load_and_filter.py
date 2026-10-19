# 01_load_and_filter.py
# Загрузка и фильтрация DSLIB

import pandas as pd
import numpy as np
from pathlib import Path

# === 1. ПУТИ ===
INPUT_FILE = r"C:\Users\z.davudov\Documents\Диплом\DSLIB3.4\DSLIB 3.4\DSLIB_Analysis_Sheet.xlsx"
SHEET_NAME = "DSLIB"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# === 2. ЗАГРУЗКА ===
# Пропускаем первые 2 строки (заголовки многоуровневые)
# Реальные названия колонок — в строке 3 (индекс 2)
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=2)

print("=" * 60)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 60)
print(f"Всего строк: {len(df)}")
print(f"Всего колонок: {len(df.columns)}")
print(f"\nПервые 20 колонок:\n{list(df.columns[:20])}")

# === 3. ОЧИСТКА ===
# Заменяем мусорные значения на NaN
def clean_cell(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip()
        if s in ["-", "N/A", "n/a", "NA", ""]:
            return np.nan
    return x

try:
    df = df.map(clean_cell)
except AttributeError:
    df = df.applymap(clean_cell)

# Убираем строки, где Code пустой (это либо итоги, либо мусор)
df = df[df["Code"].notna()].copy()
df = df[df["Code"].astype(str).str.startswith("C")].copy()

print(f"\nПосле удаления пустых строк: {len(df)}")

# === 4. ТИПЫ ДАННЫХ ===
# Числовые колонки
numeric_cols = [
    "# activities", "PD (days)", "BAC",
    "Early/late", "Under/over budget",
    "SP", "AD", "LA", "TF", "RI",
    "Real Duration", "Real Cost",
]

# Приводим к числам (что не парсится — станет NaN)
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# === 5. ФИЛЬТРАЦИЯ ===
# 5.1. Все Construction
df_construction = df[df["Sector"].astype(str).str.contains("Construction", case=False, na=False)].copy()
print(f"\nConstruction (все подвиды): {len(df_construction)}")

# 5.2. Группа А: Construction + Real Duration заполнен
df_group_a = df_construction[df_construction["Real Duration"].notna()].copy()
print(f"Группа А (Construction + Real Duration): {len(df_group_a)}")

# 5.3. Группа с tracking G/Y
df_tracking = df_construction[df_construction["Tracking"].astype(str).str.strip().isin(["G", "Y"])].copy()
print(f"Construction + Tracking G/Y: {len(df_tracking)}")

# 5.4. Группа А + Tracking G/Y (для EVM-модели)
df_group_a_evm = df_group_a[df_group_a["Tracking"].astype(str).str.strip().isin(["G", "Y"])].copy()
print(f"Группа А + Tracking G/Y (для EVM): {len(df_group_a_evm)}")

# === 6. РАСПРЕДЕЛЕНИЕ ПО ПОДВИДАМ ===
print("\n" + "=" * 60)
print("РАСПРЕДЕЛЕНИЕ ПО ПОДВИДАМ CONSTRUCTION (группа А)")
print("=" * 60)
print(df_group_a["Sector"].value_counts())

# === 7. ПРОВЕРКА EVM-МЕТРИК ===
evm_cols = ["CPI", "SPI", "SPI(t)", "p-factor"]
print("\n" + "=" * 60)
print("ЗАПОЛНЕННОСТЬ EVM-МЕТРИК В ГРУППЕ А")
print("=" * 60)
for col in evm_cols:
    if col in df_group_a.columns:
        # Для avg-версий
        filled = df_group_a[col].notna().sum()
        print(f"{col}: {filled} / {len(df_group_a)}")

# === 8. СОХРАНЕНИЕ ===
df_group_a.to_csv(OUTPUT_DIR / "group_a.csv", index=False)
df_group_a_evm.to_csv(OUTPUT_DIR / "group_a_evm.csv", index=False)
df_construction.to_csv(OUTPUT_DIR / "construction_all.csv", index=False)
df_tracking.to_csv(OUTPUT_DIR / "construction_tracking.csv", index=False)

print("\n" + "=" * 60)
print("ФАЙЛЫ СОХРАНЕНЫ")
print("=" * 60)
print(f"group_a.csv             — {len(df_group_a)} проектов")
print(f"group_a_evm.csv         — {len(df_group_a_evm)} проектов")
print(f"construction_all.csv    — {len(df_construction)} проектов")
print(f"construction_tracking.csv — {len(df_tracking)} проектов")

# === 9. КРАТКАЯ СТАТИСТИКА ===
print("\n" + "=" * 60)
print("СТАТИСТИКА ПО ГРУППЕ А")
print("=" * 60)
print(df_group_a[["# activities", "PD (days)", "Real Duration", "Early/late"]].describe())