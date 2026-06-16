import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks

"""Класс предобработки данных для обнаружения выбросов в датасете."""


class DataPreprocessing:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def select_experiment(self, speed, load):
        return self.df[
            (self.df["speedSet"] == speed) & (self.df["load_value"] == load)
        ].copy()

    # IQR метод
    def detect_outliers_iqr(self, columns=None, multiplier=1.5):

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns

        outlier_mask = pd.Series(False, index=self.df.index)

        for column in columns:
            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

            mask = (self.df[column] < lower) | (self.df[column] > upper)

            outlier_mask |= mask

        return self.df[outlier_mask]

    # Z-score метод
    def detect_outliers_zscore(self, columns=None, threshold=3):

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns

        z_scores = np.abs(zscore(self.df[columns], nan_policy="omit"))

        mask = (z_scores > threshold).any(axis=1)

        return self.df[mask]

    # Метод Изоляционного леса
    def detect_outliers_isolation_forest(
        self, columns=None, contamination=0.01, random_state=42
    ):

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns

        model = IsolationForest(contamination=contamination, random_state=random_state)

        predictions = model.fit_predict(self.df[columns])

        mask = predictions == -1

        return self.df[mask]

    # LOF-метод
    def detect_outliers_lof(self, columns=None, n_neighbors=20, contamination=0.01):

        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns

        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

        predictions = model.fit_predict(self.df[columns])

        mask = predictions == -1

        return self.df[mask]

    def compare_outlier_methods(self):

        results = []

        methods = {
            "IQR": self.detect_outliers_iqr,
            "Z-score": self.detect_outliers_zscore,
            "Isolation Forest": self.detect_outliers_isolation_forest,
            "LOF": self.detect_outliers_lof,
        }

        for method_name, method in methods.items():
            sensor1_outliers = method(columns=["sensor1"])
            sensor2_outliers = method(columns=["sensor2"])

            sensor1_count = len(sensor1_outliers)
            sensor2_count = len(sensor2_outliers)

            combined_count = len(
                pd.Index(sensor1_outliers.index).union(sensor2_outliers.index)
            )

            results.append([method_name, sensor1_count, sensor2_count, combined_count])

        return pd.DataFrame(
            results, columns=["Method", "Sensor1", "Sensor2", "Combined"]
        )

    def analyze_outliers(
        self, column, neighbour_window=5, min_methods=2, deviation_threshold=2
    ):

        iqr_idx = set(self.detect_outliers_iqr([column]).index)
        z_idx = set(self.detect_outliers_zscore([column]).index)
        if_idx = set(self.detect_outliers_isolation_forest([column]).index)
        lof_idx = set(self.detect_outliers_lof([column]).index)

        all_indexes = iqr_idx | z_idx | if_idx | lof_idx

        results = []

        for idx in all_indexes:
            # Условие 1. Количество голосов

            votes = sum([idx in iqr_idx, idx in z_idx, idx in if_idx, idx in lof_idx])

            # Условие 2. Анализ соседей

            start = max(0, idx - neighbour_window)
            end = min(len(self.df), idx + neighbour_window + 1)

            neighbours = (
                self.df[column].iloc[start:end].drop(index=idx, errors="ignore")
            )

            local_mean = neighbours.mean()
            local_std = neighbours.std()
            local_median = neighbours.median()
            deviation_percent = (
                abs(self.df.loc[idx, column] - local_median) / abs(local_median) * 100
            )

            if local_std == 0:
                continue

            is_sharp_outlier = (
                abs(self.df.loc[idx, column] - local_mean)
                > deviation_threshold * local_std
            )

            if votes >= min_methods and is_sharp_outlier:
                results.append(
                    {
                        "index": idx,
                        "value": self.df.loc[idx, column],
                        "votes": votes,
                        "local_mean": local_mean,
                        "local_median": local_median,
                        "Deviation, %": deviation_percent,
                    }
                )

        return pd.DataFrame(results)

    def replace_confirmed_outliers(self, column, analyzed_outliers):

        for _, row in analyzed_outliers.iterrows():
            idx = row["index"]
            self.df.loc[idx, column] = row["local_median"]

        return self.df


class DataVisualising:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_outlier_method_comparison(self, comparison_df):

        methods = comparison_df["Method"]

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width, comparison_df["Sensor1"], width, label="Сенсор 1")
        bars2 = ax.bar(x, comparison_df["Sensor2"], width, label="Сенсор 2")
        bars3 = ax.bar(
            x + width,
            comparison_df["Combined"],
            width,
            label="Комбинированные",
        )

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )

        ax.set_title("Сравнение методов поиска выбросов")
        ax.set_xlabel("Метод")
        ax.set_ylabel("Количество обнаруженных выбросов")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_data_overview(self, column):

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

        # Верхний график на всю ширину
        ax1 = fig.add_subplot(gs[0, :])

        # Нижние графики
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        # График 1. Временной ряд
        ax1.plot(self.df.index, self.df[column])
        ax1.set_title(f"Временной ряд ({column})")
        ax1.set_xlabel("Номер наблюдения")
        ax1.set_ylabel(column)
        ax1.grid(True)

        # График 2. BoxPlot
        ax2.boxplot(self.df[column])
        ax2.set_title(f"BoxPlot ({column})")
        ax2.set_ylabel(column)

        # График 3. Гистограмма
        ax3.hist(self.df[column], bins=50)
        ax3.set_title(f"Гистограмма ({column})")
        ax3.set_xlabel(column)
        ax3.set_ylabel("Частота")

        plt.tight_layout()
        plt.show()

    def plot_acf(self, lags, acf_values, period):

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(lags, acf_values)
        ax.axvline(period, color="red", linestyle="--", label=f"Период = {period}")
        ax.set_title("Автокорреляционная функция")
        ax.set_xlabel("Лаг")
        ax.set_ylabel("Корреляция")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_fft_spectrum(self, frequencies, amplitudes, dominant_frequency):

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(frequencies, amplitudes)
        ax.axvline(
            dominant_frequency,
            color="red",
            linestyle="--",
            label=f"Частота = {dominant_frequency:.4f} Гц",
        )

        ax.set_title("Спектр сигнала (FFT)")
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("Амплитуда")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()


class PeriodicityAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def find_period_acf(
        self, column, expected_period=None, search_window=50, nlags=1000
    ):

        signal = self.df[column].values
        # Удаляем среднее значение
        signal = signal - np.mean(signal)
        # Считаем ACF
        acf_values = acf(signal, nlags=nlags, fft=True)
        # Ищем пики только в нужном диапазоне лагов
        peaks, properties = find_peaks(
            acf_values,
            prominence=0.01,  # можно менять 0.005-0.05
        )

        candidate_peaks = peaks

        if expected_period is not None:
            mask = (peaks >= expected_period - search_window) & (
                peaks <= expected_period + search_window
            )
            candidate_peaks = peaks[mask]

        if len(candidate_peaks) > 0:
            period = candidate_peaks[np.argmax(acf_values[candidate_peaks])]
        else:
            period = expected_period

        lags = np.arange(len(acf_values))

        return period, lags, acf_values, peaks

    def find_period_fft(self, column, sampling_interval=0.0002):

        signal = self.df[column].values
        n = len(signal)

        signal = signal - np.mean(signal)
        # проверка
        print(signal.min())
        print(signal.max())
        print(signal.std())
        print(
            f"mean={np.mean(self.df[column]):.6f}, "
            f"std={np.std(self.df[column]):.6f}, "
            f"min={np.min(self.df[column]):.6f}, "
            f"max={np.max(self.df[column]):.6f}"
        )

        yf = fft(signal)
        xf = fftfreq(n, sampling_interval)

        positive = xf > 0

        frequencies = xf[positive]
        amplitudes = 2.0 / n * np.abs(yf[positive])
        # peak_idx = np.argmax(amplitudes)
        # dominant_frequency = frequencies[peak_idx]

        mask = (frequencies > 15) & (frequencies < 35)

        freq_local = frequencies[mask]
        amp_local = amplitudes[mask]

        top_idx = np.argsort(amp_local)[-5:]
        print("\n5 наиболее мощных спектральных пиков FFT:")
        for idx in reversed(top_idx):
            freq = freq_local[idx]
            period = int(round(1 / (freq * sampling_interval)))
            print(f"freq={freq:.4f}, period={period}, amp={amp_local[idx]:.5f}")

        best_idx = top_idx[-1]

        dominant_frequency = freq_local[best_idx]

        period_seconds = 1 / dominant_frequency
        period_samples = int(round(period_seconds / sampling_interval))

        return (
            period_samples,
            frequencies,
            amplitudes,
            dominant_frequency,
        )


def main():

    df = pd.read_csv("Datasets/no_fault.csv")

    subset = df[(df["speedSet"] == 25) & (df["load_value"] == 0)].copy()
    subset = subset[["sensor1", "sensor2", "time_x"]]
    subset["time_x"] = pd.to_datetime(subset["time_x"])
    subset = subset.reset_index(drop=True)

    periodicity = PeriodicityAnalyzer(subset)
    visualizer = DataVisualising(subset)

    sensors = ["sensor1", "sensor2"]
    results = []

    for sensor in sensors:
        print(f"\n{'=' * 50}")
        print(f"Анализ {sensor}")
        print(f"{'=' * 50}")

        sensor_preprocessor = DataPreprocessing(subset.copy())

        # 1. Поиск периодичности ДО очистки

        period_fft, frequencies, amplitudes, dominant_frequency = (
            periodicity.find_period_fft(sensor)
        )

        print(f"\nПериод FFT: {period_fft}")
        visualizer.plot_fft_spectrum(frequencies, amplitudes, dominant_frequency)

        period_acf, lags, acf_values, peaks = periodicity.find_period_acf(
            sensor, expected_period=period_fft, search_window=int(period_fft * 0.25)
        )

        print(f"Период ACF: {period_acf}\n")
        visualizer.plot_acf(lags, acf_values, period_acf)

        # 2. Статистика выбросов

        comparison = sensor_preprocessor.compare_outlier_methods()
        visualizer.plot_data_overview(column=sensor)
        print("Количество выбросов, выявленных различными методами: ")
        print(comparison)
        visualizer.plot_outlier_method_comparison(comparison)

        export_choice = input(
            "\nВыгрузить подробную информацию о выбросах в CSV? (y/n): "
        ).lower()
        export_outliers = export_choice == "y"

        # 3. Анализ подтвержденных выбросов и выгрузка(доп)

        confirmed_outliers = sensor_preprocessor.analyze_outliers(column=sensor)
        print("\nПодтвержденные выбросы(первые 5):")
        print(confirmed_outliers.head())
        print(f"\nПодтверждено выбросов: {len(confirmed_outliers)}")

        if export_outliers:
            filename = f"Выбросы_{sensor}.csv"
            confirmed_outliers.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"Файл сохранен: {filename}")

        # 4. Очистка

        clean_df = sensor_preprocessor.replace_confirmed_outliers(
            column=sensor, analyzed_outliers=confirmed_outliers
        )

        # 5. Повторный поиск периода

        period_test = PeriodicityAnalyzer(clean_df)
        period_fft_after, _, _, _ = period_test.find_period_fft(column=sensor)
        print(f"\nПериод FFT после очистки: {period_fft_after}")

        # 6. Итоговая таблица

        results.append(
            {
                "Сенсор": sensor,
                "Период до очистки": period_fft,
                "Подтверждено выбросов": len(confirmed_outliers),
                "Доля выбросов, %": round(
                    len(confirmed_outliers) / len(subset) * 100, 2
                ),
                "Период после очистки": period_fft_after,
            }
        )
    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print(f"{'=' * 70}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
