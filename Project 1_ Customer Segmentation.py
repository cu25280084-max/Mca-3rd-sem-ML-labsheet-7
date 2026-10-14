from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA_PATH)
    print("Dataset shape:", data.shape)
    print(data.head(), "\n")

    features = [
        "Age", "AnnualIncome_kUSD", "SpendingScore",
        "PurchaseFrequency", "AveragePurchaseValue_USD", "TotalSpend_USD"
    ]
    X = data[features].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select k using silhouette score.
    scores = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(scores, key=scores.get)
    print("K-Means silhouette scores:", {k: round(v, 4) for k, v in scores.items()})
    print("Selected number of clusters:", best_k)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data["KMeansCluster"] = kmeans.fit_predict(X_scaled)

    # Compare with hierarchical clustering using the same k.
    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    data["HierarchicalCluster"] = hierarchical.fit_predict(X_scaled)

    km_sil = silhouette_score(X_scaled, data["KMeansCluster"])
    hc_sil = silhouette_score(X_scaled, data["HierarchicalCluster"])
    print(f"K-Means silhouette score: {km_sil:.4f}")
    print(f"Hierarchical silhouette score: {hc_sil:.4f}")

    # PCA is used only to visualize the high-dimensional scaled features.
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    data["PCA1"], data["PCA2"] = components[:, 0], components[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.title("K-Means: Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.xticks(list(scores.keys()))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=data, x="PCA1", y="PCA2", hue="KMeansCluster", palette="tab10", s=55)
    plt.title("Customer Segmentation (K-Means, PCA view)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_clusters_pca.png", dpi=160)
    plt.close()

    summary = data.groupby("KMeansCluster")[features].mean().round(2)
    print("\nCustomer profile by K-Means cluster:\n", summary)
    summary.to_csv(OUTPUT_DIR / "cluster_profiles.csv")
    data.to_csv(OUTPUT_DIR / "customer_clusters.csv", index=False)

    comparison = pd.DataFrame({
        "Method": ["K-Means", "Hierarchical"],
        "SilhouetteScore": [km_sil, hc_sil]
    })
    comparison.to_csv(OUTPUT_DIR / "method_comparison.csv", index=False)
    print("\nSaved results and plots in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
