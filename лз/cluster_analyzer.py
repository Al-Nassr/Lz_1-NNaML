import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class ClusterAnalyzer:
    """Инструмент подбора числа кластеров, разбиения данных K-Means и визуализации."""

    def __init__(self, k_range=range(2, 16), random_state=42):
        self.k_grid = sorted(set(int(k) for k in k_range))
        if len(self.k_grid) < 2 or self.k_grid[0] < 2:
            raise ValueError("Диапазон k должен содержать минимум два целых значения >= 2.")
        self.random_state = random_state
        self.best_k = None
        self.model = None
        self.labels = None
        self.metrics_df = None
        self.dataset_name = "dataset.csv"

    def load_data(self, filepath="dataset.csv"):
        """Загружает CSV-файл и выделяет матрицу числовых признаков."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Файл не найден: {os.path.abspath(filepath)}")

        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("Переданный CSV-файл не содержит строк.")

        cols = [col for col in df.columns if str(col).startswith("feature_")]
        if not cols:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            raise ValueError("В таблице отсутствуют числовые колонки признаков.")

        X = df[cols].to_numpy(dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("В матрице признаков обнаружены пропуски (NaN).")

        self.dataset_name = os.path.basename(filepath)
        print(f"[*] Загружено {X.shape[0]} объектов, признаков: {X.shape[1]} ({', '.join(cols)})")
        return df, X

    @staticmethod
    def _find_elbow_point(k_array, wcss_array):
        """Находит точку излома кривой WCSS по максимальному расстоянию до хорды."""
        xs = np.asarray(k_array, dtype=np.float64)
        ys = np.asarray(wcss_array, dtype=np.float64)

        dx = xs[-1] - xs[0]
        dy = ys[0] - ys[-1]
        if dx == 0.0 or dy == 0.0:
            return int(xs[0]), np.zeros_like(xs)

        x_scaled = (xs - xs[0]) / dx
        y_scaled = (ys - ys[-1]) / dy

        # Перпендикулярное расстояние до прямой, соединяющей (0, 1) и (1, 0)
        chord_dist = (1.0 - x_scaled - y_scaled) / np.sqrt(2.0)
        elbow_idx = int(np.argmax(chord_dist))
        return int(xs[elbow_idx]), chord_dist

    def find_optimal_k(self, X):
        """Перебирает значения k, считает метрики и определяет оптимум по методу локтя."""
        records = []
        sil_sample = min(10000, len(X))

        for k in self.k_grid:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=self.random_state,
            )
            preds = km.fit_predict(X)
            sil = silhouette_score(X, preds, sample_size=sil_sample, random_state=self.random_state)
            ch = calinski_harabasz_score(X, preds)

            records.append({
                "k": k,
                "inertia": km.inertia_,
                "silhouette": sil,
                "calinski_harabasz": ch,
            })

        df_metrics = pd.DataFrame(records)
        wcss_vals = df_metrics["inertia"].to_numpy()
        df_metrics["inertia_drop"] = np.concatenate(([np.nan], -np.diff(wcss_vals)))

        best_k, elbow_scores = self._find_elbow_point(self.k_grid, wcss_vals)
        df_metrics["elbow_score"] = elbow_scores

        self.metrics_df = df_metrics
        self.best_k = best_k

        best_row = df_metrics.loc[df_metrics["k"] == best_k].iloc[0]
        print("\n--- Метрики качества кластеризации ---")
        print(df_metrics.to_string(index=False, float_format=lambda val: f"{val:.4f}"))
        print(
            f"\n=> Оптимальное число кластеров (метод локтя): k = {best_k} "
            f"(Inertia = {best_row['inertia']:.2f}, Silhouette = {best_row['silhouette']:.4f})"
        )
        return self.best_k

    def fit_predict(self, X, n_clusters=None):
        """Обучает итоговую модель K-Means и возвращает метки кластеров."""
        k = self.best_k if n_clusters is None else int(n_clusters)
        if k is None:
            k = self.find_optimal_k(X)

        self.best_k = int(k)
        self.model = KMeans(
            n_clusters=self.best_k,
            init="k-means++",
            n_init=30,
            random_state=self.random_state,
        )
        self.labels = self.model.fit_predict(X)
        return self.labels

    def save_model(self, filepath="kmeans_model.joblib"):
        """Сохраняет обученную модель K-Means на диск."""
        if self.model is None:
            raise RuntimeError("Сначала необходимо обучить модель через fit_predict().")

        joblib.dump(self.model, filepath)
        print(f"[*] Модель сохранена: {filepath}")

    def plot_results(self, X, save_prefix="cluster_report"):
        """Строит и сохраняет графики метрик подбора k и PCA-проекции кластеров."""
        if self.metrics_df is None:
            self.find_optimal_k(X)
        if self.model is None or self.labels is None:
            self.fit_predict(X)

        try:
            plt.style.use("ggplot")
        except OSError:
            plt.style.use("default")

        # 1. Графики подбора оптимального k
        fig_m, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(13, 5))

        ks = self.metrics_df["k"].to_numpy()
        wcss_mln = self.metrics_df["inertia"].to_numpy() / 1e6
        sil_vals = self.metrics_df["silhouette"].to_numpy()
        in_grid = np.any(ks == self.best_k)

        ax_elbow.plot(ks, wcss_mln, marker="o", linewidth=2.0, color="#0f766e", label="WCSS (Inertia)")
        ax_elbow.plot(
            [ks[0], ks[-1]],
            [wcss_mln[0], wcss_mln[-1]],
            linestyle=":",
            linewidth=1.4,
            color="#475569",
            label="Базовая хорда",
        )
        if in_grid:
            opt_y = wcss_mln[ks == self.best_k][0]
            ax_elbow.scatter([self.best_k], [opt_y], color="#b91c1c", s=90, zorder=6)
        ax_elbow.axvline(
            self.best_k,
            color="#b91c1c",
            linestyle="--",
            linewidth=1.5,
            label=f"Локоть: k = {self.best_k}",
        )
        ax_elbow.set_title("Метод локтя (Inertia / WCSS)", fontsize=11.5, fontweight="bold")
        ax_elbow.set_xlabel("Количество кластеров (k)")
        ax_elbow.set_ylabel(r"WCSS ($\times 10^6$)")
        ax_elbow.set_xticks(self.k_grid)
        ax_elbow.legend(frameon=True)

        ax_sil.plot(ks, sil_vals, marker="D", linewidth=2.0, color="#6d28d9", label="Silhouette Score")
        if in_grid:
            opt_s = sil_vals[ks == self.best_k][0]
            ax_sil.scatter([self.best_k], [opt_s], color="#b91c1c", s=90, zorder=6)
            sil_lbl = f"Выбранное k = {self.best_k} ({opt_s:.4f})"
        else:
            sil_lbl = f"Выбранное k = {self.best_k}"
        ax_sil.axvline(self.best_k, color="#b91c1c", linestyle="--", linewidth=1.5, label=sil_lbl)
        ax_sil.set_title("Коэффициент силуэта (Silhouette Score)", fontsize=11.5, fontweight="bold")
        ax_sil.set_xlabel("Количество кластеров (k)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_xticks(self.k_grid)
        ax_sil.legend(frameon=True)

        fig_m.suptitle(
            f"Подбор числа кластеров (оптимум k = {self.best_k})",
            fontsize=13,
            fontweight="bold",
        )
        fig_m.tight_layout()
        metrics_path = f"{save_prefix}_metrics.png"
        fig_m.savefig(metrics_path, dpi=200)

        # 2. Визуализация разбиения (PCA 2D и 3D)
        n_features = X.shape[1]
        if n_features < 2:
            print(f"[*] График метрик сохранен: {metrics_path}")
            plt.show()
            return

        n_components = min(3, n_features)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        proj = pca.fit_transform(X)
        centers_proj = pca.transform(self.model.cluster_centers_)
        explained = pca.explained_variance_ratio_ * 100.0

        cmap = plt.get_cmap("tab10" if self.best_k <= 10 else "tab20")
        has_3d = n_features >= 3
        fig_c = plt.figure(figsize=(14, 6) if has_3d else (7.5, 6))

        ax2d = fig_c.add_subplot(1, 2, 1) if has_3d else fig_c.add_subplot(1, 1, 1)
        for cid in range(self.best_k):
            mask = self.labels == cid
            ax2d.scatter(
                proj[mask, 0],
                proj[mask, 1],
                color=cmap(cid % cmap.N),
                label=f"Кластер {cid} (n={mask.sum()})",
                alpha=0.45,
                s=12,
                rasterized=True,
            )
        ax2d.scatter(
            centers_proj[:, 0],
            centers_proj[:, 1],
            c="#facc15",
            marker="*",
            s=220,
            edgecolors="#111827",
            linewidth=1.1,
            label="Центроиды",
            zorder=10,
        )
        ax2d.set_xlabel(f"PC1 ({explained[0]:.1f}% дисперсии)")
        ax2d.set_ylabel(f"PC2 ({explained[1]:.1f}% дисперсии)")
        ax2d.set_title(f"2D-проекция (PCA {n_features}D → 2D)", fontsize=11.5, fontweight="bold")
        leg = ax2d.legend(loc="best", fontsize=8.5, frameon=True)
        for handle in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
            handle.set_sizes([55.0])

        if has_3d:
            ax3d = fig_c.add_subplot(1, 2, 2, projection="3d")
            for cid in range(self.best_k):
                mask = self.labels == cid
                ax3d.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    proj[mask, 2],
                    color=cmap(cid % cmap.N),
                    label=f"Кластер {cid}",
                    alpha=0.38,
                    s=9,
                    rasterized=True,
                )
            ax3d.scatter(
                centers_proj[:, 0],
                centers_proj[:, 1],
                centers_proj[:, 2],
                c="#facc15",
                marker="*",
                s=200,
                edgecolors="#111827",
                linewidth=1.0,
                label="Центроиды",
                zorder=10,
            )
            ax3d.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
            ax3d.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
            ax3d.set_zlabel(f"PC3 ({explained[2]:.1f}%)")
            ax3d.set_title(
                f"3D-проекция (PCA {n_features}D → 3D, Σ={explained[:3].sum():.1f}%)",
                fontsize=11.5,
                fontweight="bold",
            )

        fig_c.suptitle(
            f"Кластеризация {self.dataset_name} (k = {self.best_k})",
            fontsize=13,
            fontweight="bold",
        )
        fig_c.tight_layout()
        clusters_path = f"{save_prefix}_clusters.png"
        fig_c.savefig(clusters_path, dpi=200)

        print(f"[*] Графики сохранены: {metrics_path} и {clusters_path}")
        plt.show()