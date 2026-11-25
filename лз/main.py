from cluster_analyzer import ClusterAnalyzer

DATA_FILE = "dataset.csv"
OUTPUT_CSV = "dataset_clustered.csv"
MODEL_FILE = "kmeans_model.joblib"
CHART_PREFIX = "cluster_report"


def main():
    print("=== Запуск анализа кластеров ===")
    tool = ClusterAnalyzer(k_range=range(2, 16), random_state=42)

    # 1. Загрузка исходных данных
    df, X = tool.load_data(DATA_FILE)

    # 2. Поиск оптимального k (метод локтя) и разбиение
    best_k = tool.find_optimal_k(X)
    labels = tool.fit_predict(X, n_clusters=best_k)

    # 3. Сохранение размеченного датасета и модели
    df["cluster"] = labels
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[*] Размеченный датасет сохранен: {OUTPUT_CSV}")
    tool.save_model(MODEL_FILE)

    # 4. Построение графиков
    tool.plot_results(X, save_prefix=CHART_PREFIX)
    print("=== Готово ===")


if __name__ == "__main__":
    main()