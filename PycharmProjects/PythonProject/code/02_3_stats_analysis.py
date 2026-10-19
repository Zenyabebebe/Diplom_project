# 02_3_stats_analysis.py
# Анализ распределения PD, Duration, относительных ошибок

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "predictions_deviation.csv"
OUTPUT_DIR = BASE_DIR / "output"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ")
print("=" * 60)

# Распределение PD
print("\nПЛАНОВЫЙ СРОК (PD):")
print(f"  Среднее:  {df['PD'].mean():.1f} дней")
print(f"  Медиана:  {df['PD'].median():.1f} дней")
print(f"  Std:      {df['PD'].std():.1f}")
print(f"  Min:      {df['PD'].min():.0f}")
print(f"  P25:      {df['PD'].quantile(0.25):.1f}")
print(f"  P75:      {df['PD'].quantile(0.75):.1f}")
print(f"  Max:      {df['PD'].max():.0f}")

# Распределение Duration
print("\nФАКТИЧЕСКИЙ СРОК (Duration):")
print(f"  Среднее:  {df['Duration_real'].mean():.1f} дней")
print(f"  Медиана:  {df['Duration_real'].median():.1f} дней")
print(f"  Std:      {df['Duration_real'].std():.1f}")
print(f"  Min:      {df['Duration_real'].min():.0f}")
print(f"  Max:      {df['Duration_real'].max():.0f}")

# Относительные ошибки
df["abs_error_baseline"] = (df["Duration_real"] - df["Duration_baseline_PD"]).abs()
df["abs_error_rf"] = (df["Duration_real"] - df["Duration_rf_pred"]).abs()

df["rel_error_baseline"] = df["abs_error_baseline"] / df["Duration_real"]
df["rel_error_rf"] = df["abs_error_rf"] / df["Duration_real"]

print("\nОТНОСИТЕЛЬНАЯ ОШИБКА (относительно фактического срока):")
print("\nBaseline (Duration = PD):")
print(f"  Средняя:  {df['rel_error_baseline'].mean()*100:.1f}%")
print(f"  Медиана:  {df['rel_error_baseline'].median()*100:.1f}%")

print("\nRF (наша модель):")
print(f"  Средняя:  {df['rel_error_rf'].mean()*100:.1f}%")
print(f"  Медиана:  {df['rel_error_rf'].median()*100:.1f}%")

print(f"\n✅ Улучшение относительной ошибки: "
      f"{(df['rel_error_baseline'].mean() - df['rel_error_rf'].mean())*100:.1f} п.п.")

# Категоризация по размеру проекта
print("\n" + "=" * 60)
print("ОШИБКА ПО РАЗМЕРУ ПРОЕКТА")
print("=" * 60)

bins = [0, 100, 200, 400, 800, 10000]
labels = ["<100 дней", "100-200", "200-400", "400-800", ">800"]
df["size_category"] = pd.cut(df["PD"], bins=bins, labels=labels)

for cat in labels:
    subset = df[df["size_category"] == cat]
    if len(subset) == 0:
        continue
    print(f"\n{cat} (n={len(subset)}):")
    print(f"  Средняя PD:       {subset['PD'].mean():.0f} дней")
    print(f"  Baseline MAE:     {subset['abs_error_baseline'].mean():.1f} дней "
          f"({subset['rel_error_baseline'].mean()*100:.1f}%)")
    print(f"  RF MAE:           {subset['abs_error_rf'].mean():.1f} дней "
          f"({subset['rel_error_rf'].mean()*100:.1f}%)")

# Сохраняем
df.to_csv(OUTPUT_DIR / "stats_analysis.csv", index=False)
print(f"\n✅ Сохранено: {OUTPUT_DIR / 'stats_analysis.csv'}")