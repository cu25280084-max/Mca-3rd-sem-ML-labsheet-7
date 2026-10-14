from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "news_articles.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH)
    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")
    data["combined_text"] = data["title"] + " " + data["text"]
    print("Articles:", len(data))
    print(data[["title", "category"]].head(), "\n")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(data["combined_text"])

    # Choose k by silhouette score using cosine distance.
    scores = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        scores[k] = silhouette_score(X, labels, metric="cosine")

    best_k = max(scores, key=scores.get)
    print("K-Means cosine silhouette scores:", {k: round(v, 4) for k, v in scores.items()})
    print("Selected number of clusters:", best_k)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data["KMeansCluster"] = kmeans.fit_predict(X)

    # Agglomerative clustering uses a dense cosine-distance matrix.
    distance_matrix = cosine_distances(X)
    hierarchical = AgglomerativeClustering(
        n_clusters=best_k,
        metric="precomputed",
        linkage="average"
    )
    data["HierarchicalCluster"] = hierarchical.fit_predict(distance_matrix)

    km_sil = silhouette_score(X, data["KMeansCluster"], metric="cosine")
    hc_sil = silhouette_score(X, data["HierarchicalCluster"], metric="cosine")
    print(f"K-Means cosine silhouette score: {km_sil:.4f}")
    print(f"Hierarchical cosine silhouette score: {hc_sil:.4f}")

    # SVD creates a 2-D representation for plotting sparse TF-IDF features.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coordinates = svd.fit_transform(X)
    data["SVD1"], data["SVD2"] = coordinates[:, 0], coordinates[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.title("News Clustering: Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Cosine silhouette score")
    plt.xticks(list(scores.keys()))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="SVD1", y="SVD2", hue="KMeansCluster", palette="tab10", s=65)
    plt.title("News Article Clusters (TF-IDF + K-Means, SVD view)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "news_clusters_svd.png", dpi=160)
    plt.close()

    comparison = pd.DataFrame({
        "Method": ["K-Means", "Hierarchical"],
        "CosineSilhouetteScore": [km_sil, hc_sil]
    })
    comparison.to_csv(OUTPUT_DIR / "method_comparison.csv", index=False)
    data[["title", "category", "text", "KMeansCluster", "HierarchicalCluster"]].to_csv(
        OUTPUT_DIR / "clustered_articles.csv", index=False
    )

    print("\nSample clustered articles:")
    print(data[["title", "category", "KMeansCluster", "HierarchicalCluster"]].head(15).to_string(index=False))
    print("\nSaved results and plots in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
