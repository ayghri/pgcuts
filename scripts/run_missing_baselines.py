"""Run spectral clustering baselines for missing dataset/model combos."""
import sys, numpy as np, scipy.sparse as sp, json
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

from sklearn.cluster import SpectralClustering, KMeans
from embedata import load_embeddings
from pgcuts.graph import build_rbf_knn_graph
from pgcuts.metrics import evaluate_clustering

REPR_DIR = '/buckets/representations'

MISSING = [
    ('aircraft', 'clipvitL14'), ('aircraft', 'dinov2'), ('aircraft', 'dinov3b'),
    ('dtd', 'clipvitL14'), ('dtd', 'dinov2'), ('dtd', 'dinov3b'),
    ('flowers', 'clipvitL14'), ('flowers', 'dinov2'), ('flowers', 'dinov3b'),
    ('pets', 'clipvitL14'), ('pets', 'dinov2'), ('pets', 'dinov3b'),
    ('resisc45', 'clipvitL14'), ('resisc45', 'dinov2'), ('resisc45', 'dinov3b'),
]

results = []

for ds_name, model_name in MISSING:
    print(f'\n=== {ds_name}/{model_name} ===')
    ds = load_embeddings(ds_name, model_name, REPR_DIR, split='train')
    X = ds.feats.astype(np.float32)
    y = ds.labels
    unique = np.unique(y); lmap = {o: n for n, o in enumerate(unique)}
    y = np.array([lmap[l] for l in y])
    K = len(unique); N = X.shape[0]
    print(f'  N={N}, K={K}')

    # Build graph
    W = build_rbf_knn_graph(X, n_neighbors=min(50, N - 1))

    # Spectral clustering
    print(f'  Running spectral clustering...')
    sc = SpectralClustering(
        n_clusters=K,
        affinity='precomputed',
        assign_labels='kmeans',
        n_init=10,
        random_state=42,
    )
    sc_labels = sc.fit_predict(W)
    sc_res = evaluate_clustering(y, sc_labels, K)

    # Also run KMeans for comparison
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
    km_res = evaluate_clustering(y, km.labels_, K)

    r = {
        'dataset': ds_name, 'model': model_name, 'K': K, 'N': N,
        'sc_acc': sc_res['accuracy'], 'sc_nmi': sc_res['nmi'],
        'km_acc': km_res['accuracy'], 'km_nmi': km_res['nmi'],
    }
    results.append(r)

    print(f'  SC: acc={sc_res["accuracy"]:.4f} nmi={sc_res["nmi"]:.4f}')
    print(f'  KM: acc={km_res["accuracy"]:.4f} nmi={km_res["nmi"]:.4f}')

# Save
with open('results/missing_baselines.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print table for LaTeX
print(f'\n{"="*60}')
print(f'{"Dataset":<15} {"Model":<12} {"SC ACC":>8} {"SC NMI":>8} {"KM ACC":>8} {"KM NMI":>8}')
print('-' * 60)
for r in results:
    print(f'{r["dataset"]:<15} {r["model"]:<12} {r["sc_acc"]:>8.4f} {r["sc_nmi"]:>8.4f} {r["km_acc"]:>8.4f} {r["km_nmi"]:>8.4f}')
