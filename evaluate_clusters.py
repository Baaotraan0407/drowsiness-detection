from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from clustering import DATA_PATH, FEATURE_COLUMNS, RANDOM_STATE

OUT = Path("data/cluster_evaluation.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = StandardScaler().fit_transform(df[FEATURE_COLUMNS])

    rows = []
    for k in range(2, 10):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        rows.append({
            "k": k,
            "silhouette": round(silhouette_score(X, labels), 3),
            "ari_vs_generated_label": round(adjusted_rand_score(df["student_cluster"], labels), 3),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved to {OUT}")
    print("Silhouette near 0 = overlapping clusters; near 1 = well separated.")
    print("High ARI at k = 6 means K-Means recovers the generator's own labels (circular).")


if __name__ == "__main__":
    main()