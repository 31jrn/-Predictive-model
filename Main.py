import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


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

    # Сранение результатов разных методов
    def compare_outlier_methods(self, columns=None):

        total_rows = len(self.df)

        results = []

        methods = {
            "IQR": len(self.detect_outliers_iqr(columns)),
            "Z-score": len(self.detect_outliers_zscore(columns)),
            "Isolation Forest": len(self.detect_outliers_isolation_forest(columns)),
            "LOF": len(self.detect_outliers_lof(columns)),
        }

        for method, count in methods.items():
            results.append([method, count, round(count / total_rows * 100, 2)])
        return pd.DataFrame(results, columns=["Method", "Outliers Found", "Percent"])


class DataVisualising:
    def __init__(self, dataframe):
        self.df = dataframe

    def plot_time_series(self, column):
        fig, ax = plt.subplots()
        ax.plot(self.df[column])
        plt.show()


def main():
    df = pd.read_csv("Datasets/no_fault.csv")

    preprocessor = DataPreprocessing(df)

    subset = preprocessor.select_experiment(
        speed=25,
        load=0
    )

    subset["time_x"] = pd.to_datetime(subset["time_x"])

    

if __name__ == "__main__":
    main()
