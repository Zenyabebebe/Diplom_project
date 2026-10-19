# 01_load_and_filter.py
# Загрузка и фильтрация DSLIB

import pandas as pd
import numpy as np
from pathlib import Path

# === 1. ПУТИ ===
INPUT_FILE = r"C:\Users\z.davudov\Documents\Диплом\DSLIB3.4\DSLIB 3.4\DSLIB_Analysis_Sheet.xlsx"

# Папка для результатов (рядом со скриптом)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SHEET_NAME = "DSLIB"

# === 2. ПРОВЕРКА ФАЙЛА ===
if not Path(INPUT_FILE).exists():
    print(f"❌ Файл не найден: {INPUT_FILE}")
    exit(1)

print(f"✅ Файл найден: {INPUT_FILE}")

# === 3. ЗАГРУЗКА ===
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=2)

print("\n" + "=" * 60)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 60)
print(f"Всего строк: {len(df)}")
print(f"Всего колонок: {len(df.columns)}")

# === 4. ПОКАЗЫВАЕМ ВСЕ КОЛОНКИ ===
print("\n" + "=" * 60)
print("ВСЕ КОЛОНКИ DSLIB (номер: название)")
print("=" * 60)
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")

# === 5. ИЩЕМ КОЛОНКИ С REAL DURATION ===
# Возможные варианты названий
real_duration_candidates = [
    "Real Duration", "Real duration", "RealDuration",
    "Real", "Duration", "real_duration", "Real_Duration",
]

# Также ищем колонку, которая предпоследняя (там обычно Real Duration)
print("\n" + "=" * 60)
print("ПОИСК КОЛОНКИ REAL DURATION")
print("=" * 60)

found_col = None
for cand in real_duration_candidates:
    if cand in df.columns:
        found_col = cand
        print(f"✅ Найдена колонка: '{cand}'")
        break

if found_col is None:
    print("⚠️  Не найдена по названию. Последние 5 колонок:")
    for col in df.columns[-5:]:
        print(f"   - '{col}'")
    print("\nСмотрим, какая из них числовая и содержит большие значения (это и есть Duration):")
    for col in df.columns[-5:]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 50:
            print(f"   - '{col}': {numeric.notna().sum()} числовых, среднее={numeric.mean():.1f}")
    print("\n➡️  Возьми название из списка выше и впиши вручную в переменную REAL_DURATION_COL")
    REAL_DURATION_COL = None
else:
    REAL_DURATION_COL = found_col

# === 6. ОЧИСТКА ===
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

df = df[df["Code"].notna()].copy()
df = df[df["Code"].astype(str).str.startswith("C")].copy()

print(f"\nПосле удаления пустых строк: {len(df)}")

# === 7. ФИЛЬТРАЦИЯ ===
df_construction = df[
    df["Sector"].astype(str).str.contains("Construction", case=False, na=False)
].copy()
print(f"\nConstruction (все подвиды): {len(df_construction)}")

# Если REAL_DURATION_COL найден — фильтруем группу А
if REAL_DURATION_COL is not None:
    df_construction[REAL_DURATION_COL] = pd.to_numeric(
        df_construction[REAL_DURATION_COL], errors="coerce"
    )
    df_group_a = df_construction[df_construction[REAL_DURATION_COL].notna()].copy()
    print(f"Группа А (Construction + Real Duration): {len(df_group_a)}")

    df_tracking = df_construction[
        df_construction["Tracking"].astype(str).str.strip().isin(["G", "Y"])
    ].copy()
    print(f"Construction + Tracking G/Y: {len(df_tracking)}")

    df_group_a_evm = df_group_a[
        df_group_a["Tracking"].astype(str).str.strip().isin(["G", "Y"])
    ].copy()
    print(f"Группа А + Tracking G/Y (для EVM): {len(df_group_a_evm)}")

    # Распределение по подвидам
    print("\n" + "=" * 60)
    print("РАСПРЕДЕЛЕНИЕ ПО ПОДВИДАМ CONSTRUCTION (группа А)")
    print("=" * 60)
    print(df_group_a["Sector"].value_counts())

    # Сохранение
    df_group_a.to_csv(OUTPUT_DIR / "group_a.csv", index=False)
    df_group_a_evm.to_csv(OUTPUT_DIR / "group_a_evm.csv", index=False)
    df_construction.to_csv(OUTPUT_DIR / "construction_all.csv", index=False)
    df_tracking.to_csv(OUTPUT_DIR / "construction_tracking.csv", index=False)

    print("\n" + "=" * 60)
    print("ФАЙЛЫ СОХРАНЕНЫ В:")
    print("=" * 60)
    print(OUTPUT_DIR)
    print(f"group_a.csv                — {len(df_group_a)} проектов")
    print(f"group_a_evm.csv            — {len(df_group_a_evm)} проектов")
    print(f"construction_all.csv       — {len(df_construction)} проектов")
    print(f"construction_tracking.csv  — {len(df_tracking)} проектов")
else:
    print("\n⚠️  Дальнейшая фильтрация остановлена — впиши правильное имя колонки")
    print("   в переменную REAL_DURATION_COL и запусти снова.")