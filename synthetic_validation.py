"""
synthetic_validation.py

Проверка методов предобработки данных на синтетических данных с
известной "истиной": поиск периодичности (FFT, ACF) и поиск выбросов
(IQR, Z-score, Isolation Forest, LOF).

Логика проверки периодичности:
1. Генерируем синусоидальный сигнал + гауссовский шум с заданными
   параметрами (период, амплитуда, уровень шума, объём выборки).
2. Прогоняем через него FFT- и ACF-методы.
3. Сравниваем найденный период с истинным -> абсолютная и
   относительная ошибка.
4. Каждый сценарий повторяем много раз (Монте-Карло, разные seed).
5. Дополнительно строим кривую "ошибка vs SNR" вместо отдельных точек.

Логика проверки поиска выбросов:
1. На тот же синтетический сигнал внедряем точечные выбросы в
   известные позиции (доля выбросов задаётся в процентах: 5/10/15/20/25%).
2. Прогоняем через 4 метода детекции.
3. Т.к. истинные позиции выбросов известны, считаем не просто
   "сколько нашли", а честную матрицу ошибок: TP/FP/FN ->
   Precision/Recall/F1.
4. Каждый сценарий повторяем много раз (Монте-Карло).
5. Отдельно проверяется чувствительность Isolation Forest / LOF к
   параметру contamination при неверном значении этого параметра.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import detrend as scipy_detrend

from Main import PeriodicityAnalyzer, DataPreprocessing


# 1. Генерация синтетических данных


def generate_periodic_signal(
    n=1500,
    period=50,
    noise_std=0.3,
    amplitude=1.0,
    phase=None,
    trend_slope=0.0,
    seed=None,
):
    """
    Генерирует синусоидальный сигнал с известным периодом и аддитивным
    гауссовским шумом.

    Parameters
    ----------
    n : int
        Длина выборки (число отсчётов).
    period : float
        Истинный период в отсчётах (samples).
    noise_std : float
        Стандартное отклонение шума.
    amplitude : float
        Амплитуда синусоиды.
    phase : float or None
        Начальная фаза. Если None - случайная (важно для честности
        проверки: метод не должен "подстраиваться" под фиксированную фазу).
    trend_slope : float
        Коэффициент линейного тренда (проверка нестационарных данных).
    seed : int or None
        Seed генератора случайных чисел для воспроизводимости.

    Returns
    -------
    pd.DataFrame с единственной колонкой 'value'.
    """
    rng = np.random.default_rng(seed)

    if phase is None:
        phase = rng.uniform(0, 2 * np.pi)

    t = np.arange(n)
    clean_signal = amplitude * np.sin(2 * np.pi * t / period + phase)
    clean_signal = clean_signal + trend_slope * t

    noise = rng.normal(0, noise_std, size=n)
    values = clean_signal + noise

    return pd.DataFrame({"value": values})


def compute_snr(amplitude, noise_std):
    """
    Отношение сигнал/шум по мощности (в разах, не в дБ).
    Для синусоиды мощность сигнала = amplitude^2 / 2.
    """
    signal_power = (amplitude**2) / 2
    noise_power = noise_std**2
    return signal_power / noise_power if noise_power > 0 else np.inf


# 2. Обобщённый FFT-детектор периода
"""
В Main.py метод find_period_fft "заточен" под конкретный датчик:
использует sampling_interval в секундах и жёстко ищет пик только в
диапазоне 15-35 Гц, поэтому здесь - обобщённая
версия без привязки к конкретному частотному диапазону."""


def find_period_fft_generic(signal, min_period=4, max_period=None):
    """
    Определяет доминирующий период сигнала через FFT в отсчётах
    (samples), без ограничения на конкретный частотный диапазон.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if max_period is None:
        max_period = n // 2

    # Линейный тренд даёт огромный всплеск мощности у частоты ~0, из-за
    # которого FFT может принять тренд за "период" в тысячи отсчётов -
    # поэтому тренд убирается до преобразования Фурье.
    signal = scipy_detrend(signal, type="linear")
    signal = signal - np.mean(signal)

    yf = fft(signal)
    xf = fftfreq(n, d=1.0)

    positive = xf > 0
    frequencies = xf[positive]
    amplitudes = 2.0 / n * np.abs(yf[positive])

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    mask = (frequencies <= max_freq) & (frequencies >= min_freq)

    freq_local = frequencies[mask]
    amp_local = amplitudes[mask]

    if len(freq_local) == 0:
        return None, frequencies, amplitudes, None

    best_idx = np.argmax(amp_local)
    dominant_frequency = freq_local[best_idx]
    period_samples = int(round(1.0 / dominant_frequency))

    return period_samples, frequencies, amplitudes, dominant_frequency


# 3. Одиночная проверка + Монте-Карло повторы (период)


def evaluate_period_detection(
    period,
    noise_std,
    n=1500,
    amplitude=1.0,
    trend_slope=0.0,
    n_repeats=30,
    base_seed=0,
):
    """
    Многократный прогон FFT- и ACF-детекторов на одном сценарии
    (period, noise_std, ...) для получения средней ошибки и разброса,
    а не единичного (возможно случайно удачного) результата.

    Returns
    -------
    pd.DataFrame с ошибками по каждому повтору и методу.
    """
    records = []

    for i in range(n_repeats):
        seed = base_seed + i
        df = generate_periodic_signal(
            n=n,
            period=period,
            noise_std=noise_std,
            amplitude=amplitude,
            trend_slope=trend_slope,
            seed=seed,
        )

        fft_period, _, _, _ = find_period_fft_generic(df["value"].values)

        # ACF тоже считаем на данных без тренда - иначе тренд искажает
        # автокорреляционную функцию так же, как искажал спектр FFT.
        df_detrended = df.copy()
        df_detrended["value"] = scipy_detrend(df["value"].values, type="linear")

        analyzer = PeriodicityAnalyzer(df_detrended)
        acf_period, _, _, _ = analyzer.find_period_acf(
            "value",
            expected_period=fft_period if fft_period else period,
            search_window=max(5, int(period * 0.3)),
            nlags=min(1000, n - 1),
        )

        for method_name, found in (("FFT", fft_period), ("ACF", acf_period)):
            if found is None:
                continue

            abs_error = abs(found - period)
            rel_error = abs_error / period * 100

            records.append(
                {
                    "period_true": period,
                    "noise_std": noise_std,
                    "repeat": i,
                    "method": method_name,
                    "period_found": found,
                    "abs_error": abs_error,
                    "rel_error_%": rel_error,
                }
            )

    return pd.DataFrame(records)


# 4. Группа A: 5 сценариев проверки периодичности


def run_group_a_scenarios(n_repeats=30):
    """
    Прогоняет 5 сценариев проверки периодичности с разными периодами,
    уровнями шума и наличием тренда (нестационарность).
    """
    scenarios = [
        {
            "name": "1. База (T=50, низкий шум)",
            "period": 50,
            "noise_std": 0.1,
            "trend_slope": 0.0,
        },
        {
            "name": "2. Высокий шум (T=50)",
            "period": 50,
            "noise_std": 0.8,
            "trend_slope": 0.0,
        },
        {
            "name": "3. Короткий период (T=20)",
            "period": 20,
            "noise_std": 0.1,
            "trend_slope": 0.0,
        },
        {
            "name": "4. Длинный период (T=200)",
            "period": 200,
            "noise_std": 0.1,
            "trend_slope": 0.0,
        },
        {
            "name": "5. Тренд + средний шум (T=50)",
            "period": 50,
            "noise_std": 0.3,
            "trend_slope": 0.002,
        },
    ]

    all_results = []
    summary_rows = []

    for idx, sc in enumerate(scenarios):
        raw = evaluate_period_detection(
            period=sc["period"],
            noise_std=sc["noise_std"],
            trend_slope=sc["trend_slope"],
            n_repeats=n_repeats,
            base_seed=idx * 1000,
        )
        raw["scenario"] = sc["name"]
        all_results.append(raw)

        for method in ("FFT", "ACF"):
            subset = raw[raw["method"] == method]
            if len(subset) == 0:
                continue

            summary_rows.append(
                {
                    "Сценарий": sc["name"],
                    "Метод": method,
                    "T истинный": sc["period"],
                    "Шум (σ)": sc["noise_std"],
                    "Абс. ошибка, сред.": round(subset["abs_error"].mean(), 2),
                    "Абс. ошибка, std": round(subset["abs_error"].std(), 2),
                    "Отн. ошибка %, сред.": round(subset["rel_error_%"].mean(), 2),
                    "Отн. ошибка %, std": round(subset["rel_error_%"].std(), 2),
                    "Успешных прогонов": len(subset),
                }
            )

    detailed_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    return detailed_df, summary_df


# 5. Кривая "ошибка от SNR" (вместо отдельных точек)


def run_snr_curve(period=50, n=1500, amplitude=1.0, n_repeats=20, noise_levels=None):
    """
    Строит зависимость относительной ошибки периода от уровня шума (SNR).
    Даёт кривую деградации метода, а не разрозненные точки.
    """
    if noise_levels is None:
        noise_levels = np.linspace(0.05, 1.5, 12)

    rows = []
    for noise_std in noise_levels:
        raw = evaluate_period_detection(
            period=period,
            noise_std=noise_std,
            amplitude=amplitude,
            n_repeats=n_repeats,
            base_seed=int(noise_std * 10000),
        )
        snr = compute_snr(amplitude, noise_std)

        for method in ("FFT", "ACF"):
            subset = raw[raw["method"] == method]
            if len(subset) == 0:
                continue

            rows.append(
                {
                    "noise_std": noise_std,
                    "SNR": snr,
                    "method": method,
                    "rel_error_mean_%": subset["rel_error_%"].mean(),
                    "rel_error_std_%": subset["rel_error_%"].std(),
                }
            )

    return pd.DataFrame(rows)


# 6. Визуализация периодичности


def plot_example_signal(period=50, noise_std=0.3, n=1500, seed=1):
    df = generate_periodic_signal(n=n, period=period, noise_std=noise_std, seed=seed)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["value"])
    ax.set_title(f"Пример синтетического сигнала (T={period}, σ={noise_std}, N={n})")
    ax.set_xlabel("Номер отсчёта")
    ax.set_ylabel("Значение")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_scenario_errors(summary_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = summary_df["Сценарий"].unique()
    x = np.arange(len(scenarios))
    width = 0.35

    for i, method in enumerate(("FFT", "ACF")):
        sub = (
            summary_df[summary_df["Метод"] == method]
            .set_index("Сценарий")
            .reindex(scenarios)
        )

        means = sub["Отн. ошибка %, сред."]
        stds = sub["Отн. ошибка %, std"]
        # ошибка не может быть отрицательной - нижний ус обрезаем так,
        # чтобы mean - lower_err не уходил ниже нуля (верхний ус не трогаем)
        lower_err = np.minimum(stds, means)
        upper_err = stds

        ax.bar(
            x + i * width,
            means,
            width,
            yerr=[lower_err, upper_err],
            capsize=4,
            label=method,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Относительная ошибка периода, %")
    ax.set_title("FFT vs ACF по сценариям (среднее ± std по повторам Монте-Карло)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_snr_curve(snr_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, group in snr_df.groupby("method"):
        group = group.sort_values("SNR")
        ax.plot(group["SNR"], group["rel_error_mean_%"], marker="o", label=method)
        # ошибка не может быть отрицательной - обрезаем нижнюю границу
        # заливки нулём, даже если mean - std формально уходит в минус
        lower = np.maximum(0, group["rel_error_mean_%"] - group["rel_error_std_%"])
        upper = group["rel_error_mean_%"] + group["rel_error_std_%"]
        ax.fill_between(group["SNR"], lower, upper, alpha=0.15)

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("SNR (сигнал/шум, разы, лог. шкала)")
    ax.set_ylabel("Относительная ошибка периода, %")
    ax.set_title("Деградация точности определения периода при снижении SNR")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# 7. Внедрение выбросов с известными позициями


def inject_outliers(df, column="value", fraction=0.05, magnitude_k=6.0, seed=None):
    """
    Внедряет точечные выбросы в df[column] в случайно выбранные, но
    ИЗВЕСТНЫЕ позиции.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    fraction : float
        Доля выбросов от объёма выборки (0.05 = 5%).
    magnitude_k : float
        Во сколько раз (минимум) выброс превышает локальное std сигнала.
        Магнитуда берётся случайно в диапазоне [magnitude_k, 1.5*magnitude_k],
        чтобы выбросы не были все одинаковой высоты.
    seed : int or None

    Returns
    -------
    df_with_outliers : pd.DataFrame (копия df со внедрёнными выбросами)
    true_idx : set - индексы (в терминах df.index), куда внедрён выброс
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    n_outliers = int(round(n * fraction))

    positions = rng.choice(df.index.to_numpy(), size=n_outliers, replace=False)
    true_idx = set(positions.tolist())

    local_std = df[column].std()
    signs = rng.choice([-1.0, 1.0], size=n_outliers)
    magnitudes = rng.uniform(magnitude_k, magnitude_k * 1.5, size=n_outliers)

    df_out = df.copy()
    df_out.loc[positions, column] = (
        df_out.loc[positions, column].to_numpy() + signs * magnitudes * local_std
    )

    return df_out, true_idx


# 8. Метрики качества детекции выбросов (матрица ошибок)


def confusion_metrics(true_idx, pred_idx, total_n):
    """
    Считает TP/FP/FN/TN и Precision/Recall/F1 по известным истинным
    позициям выбросов (true_idx) и позициям, найденным методом (pred_idx).
    """
    tp = len(true_idx & pred_idx)
    fp = len(pred_idx - true_idx)
    fn = len(true_idx - pred_idx)
    tn = total_n - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# 9. Одиночная проверка + Монте-Карло повторы (выбросы)


def evaluate_outlier_detection(
    fraction,
    n=1500,
    period=50,
    noise_std=0.3,
    magnitude_k=6.0,
    n_repeats=20,
    base_seed=0,
    oracle_contamination=True,
    fixed_contamination=0.05,
):
    """
    Многократный прогон 4 методов детекции выбросов на одном сценарии
    (доля выбросов = fraction) для оценки средних Precision/Recall/F1
    и их разброса.

    oracle_contamination=True: Isolation Forest / LOF получают
    contamination = истинная доля выбросов (лучший случай - параметр
    известен точно). Если False - используется fixed_contamination
    независимо от реальной доли (реалистичный случай, когда истинная
    доля заранее не известна).
    """
    records = []

    for i in range(n_repeats):
        seed = base_seed + i
        df = generate_periodic_signal(
            n=n, period=period, noise_std=noise_std, seed=seed
        )
        df_out, true_idx = inject_outliers(
            df,
            column="value",
            fraction=fraction,
            magnitude_k=magnitude_k,
            seed=seed + 500_000,
        )

        contamination = fraction if oracle_contamination else fixed_contamination
        contamination = float(np.clip(contamination, 1e-3, 0.5))

        prep = DataPreprocessing(df_out)

        detectors = {
            "IQR": lambda: prep.detect_outliers_iqr(columns=["value"]),
            "Z-score": lambda: prep.detect_outliers_zscore(columns=["value"]),
            "Isolation Forest": lambda: prep.detect_outliers_isolation_forest(
                columns=["value"], contamination=contamination
            ),
            "LOF": lambda: prep.detect_outliers_lof(
                columns=["value"], contamination=contamination
            ),
        }

        for name, detect_fn in detectors.items():
            pred_idx = set(detect_fn().index)
            metrics = confusion_metrics(true_idx, pred_idx, total_n=n)

            records.append(
                {
                    "fraction": fraction,
                    "repeat": i,
                    "method": name,
                    "n_true_outliers": len(true_idx),
                    "n_found": len(pred_idx),
                    "detected_fraction_%": len(pred_idx) / n * 100,
                    **metrics,
                }
            )

    return pd.DataFrame(records)


# 10. Группа B: 5 сценариев (доли выбросов 5/10/15/20/25%)


def run_group_b_scenarios(n_repeats=20, fractions=(0.05, 0.10, 0.15, 0.20, 0.25)):
    """
    Прогоняет 5 сценариев проверки поиска выбросов с разной долей
    выбросов в выборке (группа B из ТЗ).
    """
    all_results = []
    summary_rows = []

    for idx, fraction in enumerate(fractions):
        raw = evaluate_outlier_detection(
            fraction=fraction,
            n_repeats=n_repeats,
            base_seed=idx * 1000,
        )
        all_results.append(raw)

        for method in ("IQR", "Z-score", "Isolation Forest", "LOF"):
            subset = raw[raw["method"] == method]
            if len(subset) == 0:
                continue

            summary_rows.append(
                {
                    "Доля выбросов, %": fraction * 100,
                    "Метод": method,
                    "TP, сред.": round(subset["TP"].mean(), 1),
                    "FP, сред.": round(subset["FP"].mean(), 1),
                    "FN, сред.": round(subset["FN"].mean(), 1),
                    "Precision, сред.": round(subset["precision"].mean(), 3),
                    "Precision, std": round(subset["precision"].std(), 3),
                    "Recall, сред.": round(subset["recall"].mean(), 3),
                    "Recall, std": round(subset["recall"].std(), 3),
                    "F1, сред.": round(subset["f1"].mean(), 3),
                    "F1, std": round(subset["f1"].std(), 3),
                    "Найдено, % от выборки (сред.)": round(
                        subset["detected_fraction_%"].mean(), 2
                    ),
                }
            )

    detailed_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    return detailed_df, summary_df


def print_group_b_summary_by_method(summary_df):
    """
    Печатает не одну общую простыню, а отдельную табличку на каждый
    метод (по всем долям выбросов) - так результаты не сваливаются
    в кучу и сразу видно, как конкретный метод ведёт себя при росте
    доли выбросов.
    """
    for method in ("IQR", "Z-score", "Isolation Forest", "LOF"):
        sub = summary_df[summary_df["Метод"] == method].drop(columns=["Метод"])
        sub = sub.sort_values("Доля выбросов, %")

        print(f"\n--- {method} ---")
        print(sub.to_string(index=False))


# 10а. Чувствительность к магнитуде выброса (граница обнаружения)


def run_magnitude_sensitivity(
    fraction=0.10,
    magnitude_values=(1.5, 2.0, 3.0, 4.0, 6.0, 9.0),
    n_repeats=20,
):
    """
    В отличие от run_group_b_scenarios (где магнитуда выбросов всегда
    большая, 6-9 сигма - "лёгкий" случай), здесь магнитуда сама
    варьируется при фиксированной доле выбросов. Показывает границу,
    после которой методы перестают отличать выброс от обычного
    отклонения сигнала - то есть насколько узкий тест был бы, если
    ограничиться только группой B.
    """
    rows = []

    for magnitude_k in magnitude_values:
        raw = evaluate_outlier_detection(
            fraction=fraction,
            magnitude_k=magnitude_k,
            n_repeats=n_repeats,
            base_seed=int(magnitude_k * 10_000),
        )

        for method in ("IQR", "Z-score", "Isolation Forest", "LOF"):
            subset = raw[raw["method"] == method]
            if len(subset) == 0:
                continue

            rows.append(
                {
                    "magnitude_k (σ)": magnitude_k,
                    "method": method,
                    "precision_mean": subset["precision"].mean(),
                    "recall_mean": subset["recall"].mean(),
                    "f1_mean": subset["f1"].mean(),
                }
            )

    return pd.DataFrame(rows)


def plot_magnitude_sensitivity(magnitude_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, group in magnitude_df.groupby("method"):
        group = group.sort_values("magnitude_k (σ)")
        ax.plot(group["magnitude_k (σ)"], group["f1_mean"], marker="o", label=method)

    ax.set_xlabel("Магнитуда выброса, во сколько раз больше σ сигнала")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Граница обнаружения: F1 в зависимости от 'заметности' выброса\n"
        "(при фиксированной доле выбросов 10%)"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# 11. Чувствительность к параметру contamination (IF / LOF)


def run_contamination_sensitivity(
    true_fraction=0.10,
    contamination_values=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
    n_repeats=20,
):
    """
    Показывает, что происходит с качеством Isolation Forest / LOF,
    когда параметр contamination НЕ совпадает с реальной долей
    выбросов (истинная доля фиксирована = true_fraction, параметр
    contamination перебирается отдельно). В реальных данных истинная
    доля заранее не известна, и её приходится угадывать - этот
    эксперимент показывает цену ошибки в этой "угадайке".
    """
    rows = []

    for contamination in contamination_values:
        raw = evaluate_outlier_detection(
            fraction=true_fraction,
            n_repeats=n_repeats,
            base_seed=int(contamination * 100_000),
            oracle_contamination=False,
            fixed_contamination=contamination,
        )

        for method in ("Isolation Forest", "LOF"):
            subset = raw[raw["method"] == method]
            if len(subset) == 0:
                continue

            rows.append(
                {
                    "contamination_param": contamination,
                    "true_fraction": true_fraction,
                    "method": method,
                    "precision_mean": subset["precision"].mean(),
                    "recall_mean": subset["recall"].mean(),
                    "f1_mean": subset["f1"].mean(),
                }
            )

    return pd.DataFrame(rows)


# 12. Визуализация выбросов


def plot_example_signal_with_outliers(fraction=0.10, seed=1):
    df = generate_periodic_signal(n=1500, period=50, noise_std=0.3, seed=seed)
    df_out, true_idx = inject_outliers(df, fraction=fraction, seed=seed + 500_000)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_out.index, df_out["value"], linewidth=0.8, label="Сигнал с выбросами")
    outlier_positions = sorted(true_idx)
    ax.scatter(
        outlier_positions,
        df_out.loc[outlier_positions, "value"],
        color="red",
        zorder=5,
        label=f"Внедрённые выбросы ({len(true_idx)} шт., {fraction * 100:.0f}%)",
    )
    ax.set_title("Пример синтетических данных с внедрёнными выбросами")
    ax.set_xlabel("Номер отсчёта")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_metrics_vs_fraction(summary_df):
    """
    Три графика (Precision, Recall, F1) в зависимости от доли выбросов,
    по одной линии на метод.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metric_cols = [
        ("Precision, сред.", "Precision, std", "Precision"),
        ("Recall, сред.", "Recall, std", "Recall"),
        ("F1, сред.", "F1, std", "F1-score"),
    ]

    for ax, (mean_col, std_col, title) in zip(axes, metric_cols):
        for method, group in summary_df.groupby("Метод"):
            group = group.sort_values("Доля выбросов, %")
            ax.plot(
                group["Доля выбросов, %"], group[mean_col], marker="o", label=method
            )
            lower = np.maximum(0, group[mean_col] - group[std_col])
            upper = np.minimum(1, group[mean_col] + group[std_col])
            ax.fill_between(group["Доля выбросов, %"], lower, upper, alpha=0.12)

        ax.set_title(title)
        ax.set_xlabel("Доля выбросов в выборке, %")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Значение метрики")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Качество детекции выбросов в зависимости от их доли в выборке")
    plt.tight_layout()
    plt.show()


def plot_detected_vs_true_fraction(summary_df):
    """
    Показывает СМЕЩЕНИЕ (найдено - внедрено, в п.п.), а не абсолютные
    значения. При абсолютных значениях методы, которые находят долю
    почти идеально точно (IQR, Isolation Forest), ложатся прямо на
    линию идеала и друг на друга и визуально пропадают - в координатах
    смещения линия идеала превращается в горизонтальный ноль, и все
    методы видно раздельно, включая почти точные.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhline(0, color="black", linestyle="--", label="Идеал (найдено = внедрено)")

    for method, group in summary_df.groupby("Метод"):
        group = group.sort_values("Доля выбросов, %")
        bias = group["Найдено, % от выборки (сред.)"] - group["Доля выбросов, %"]
        ax.plot(group["Доля выбросов, %"], bias, marker="o", label=method)

    ax.set_xlabel("Истинная доля выбросов, %")
    ax.set_ylabel("Смещение (найдено - внедрено), п.п.")
    ax.set_title("Смещение оценки доли выбросов относительно истинной")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_contamination_sensitivity(sensitivity_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    true_fraction_pct = sensitivity_df["true_fraction"].iloc[0] * 100

    for method, group in sensitivity_df.groupby("method"):
        group = group.sort_values("contamination_param")
        ax.plot(
            group["contamination_param"] * 100,
            group["f1_mean"],
            marker="o",
            label=method,
        )

    ax.axvline(
        true_fraction_pct,
        color="gray",
        linestyle="--",
        label=f"Истинная доля выбросов ({true_fraction_pct:.0f}%)",
    )
    ax.set_xlabel("Параметр contamination, %")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Чувствительность Isolation Forest / LOF к ошибке в параметре contamination"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# 13. Точка входа


def main():
    print("=" * 70)
    print("ПРОВЕРКА МЕТОДОВ ПРЕДОБРАБОТКИ НА СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("=" * 70)

    # --- Периодичность ---
    plot_example_signal()

    detailed_period_df, summary_period_df = run_group_a_scenarios(n_repeats=30)
    print()
    print(summary_period_df.to_string(index=False))
    plot_scenario_errors(summary_period_df)

    snr_df = run_snr_curve()
    plot_snr_curve(snr_df)

    detailed_period_df.to_csv(
        "period_validation_detailed.csv", index=False, encoding="utf-8-sig"
    )
    summary_period_df.to_csv(
        "period_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    snr_df.to_csv("period_validation_snr_curve.csv", index=False, encoding="utf-8-sig")

    # --- Выбросы ---
    plot_example_signal_with_outliers()

    detailed_outliers_df, summary_outliers_df = run_group_b_scenarios(n_repeats=20)
    print_group_b_summary_by_method(summary_outliers_df)
    plot_metrics_vs_fraction(summary_outliers_df)
    plot_detected_vs_true_fraction(summary_outliers_df)

    sensitivity_df = run_contamination_sensitivity()
    plot_contamination_sensitivity(sensitivity_df)

    magnitude_df = run_magnitude_sensitivity()
    plot_magnitude_sensitivity(magnitude_df)

    detailed_outliers_df.to_csv(
        "outlier_validation_detailed.csv", index=False, encoding="utf-8-sig"
    )
    summary_outliers_df.to_csv(
        "outlier_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity_df.to_csv(
        "outlier_validation_contamination_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    magnitude_df.to_csv(
        "outlier_validation_magnitude_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nРезультаты сохранены: "
        "period_validation_detailed.csv, period_validation_summary.csv, "
        "period_validation_snr_curve.csv, outlier_validation_detailed.csv, "
        "outlier_validation_summary.csv, outlier_validation_contamination_sensitivity.csv, "
        "outlier_validation_magnitude_sensitivity.csv"
    )


if __name__ == "__main__":
    main()
