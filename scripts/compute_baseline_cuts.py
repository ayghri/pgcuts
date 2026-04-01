"""Compute RCut and NCut for KMeans and Spectral baselines."""
import sys, json, numpy as np, scipy.sparse as sp
sys.stdout.reconfigure(line_buffering=True)

from sklearn.cluster import KMeans, SpectralClustering
from embedata import load_embeddings
from pgcuts.graph import build_rbf_knn_graph
from pgcuts.metrics import evaluate_clustering, compute_rcut_ncut

REPR_DIR = '/buckets/representations'

datasets = ['cifar10','cifar100','stl10','aircraft','eurosat','dtd','flowers',
            'pets','food101','gtsrb','fashionmnist','mnist','imagenette','cub','resisc45']
models = ['clipvitL14', 'dinov2', 'dinov3b']

results = []

for ds in datasets:
    for model in models:
        print(f'{ds}/{model}...', end=' ', flush=True)
        try:
            d = load_embeddings(ds, model, REPR_DIR, split='train')
        except:
            print('SKIP')
            continue

        X = d.feats.astype(np.float32)
        y = d.labels
        u = np.unique(y); lmap = {o: n for n, o in enumerate(u)}
        y = np.array([lmap[l] for l in y])
        K = len(u); N = X.shape[0]

        # Subsample large datasets
        if N > 80000:
            rng = np.random.RandomState(42)
            idx = []
            for k in range(K):
                ci = np.where(y == k)[0]
                idx.append(rng.choice(ci, max(1, int(80000 * len(ci) / N)), replace=False))
            idx = np.concatenate(idx)
            X, y = X[idx], y[idx]
            N = len(y)

        W = build_rbf_knn_graph(X, n_neighbors=min(50, N - 1))

        # KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
        km_res = evaluate_clustering(y, km.labels_, K)
        km_rcut, km_ncut = compute_rcut_ncut(W, km.labels_)

        # Spectral
        sc = SpectralClustering(n_clusters=K, affinity='precomputed',
                                assign_labels='kmeans', n_init=10, random_state=42)
        sc_labels = sc.fit_predict(W)
        sc_res = evaluate_clustering(y, sc_labels, K)
        sc_rcut, sc_ncut = compute_rcut_ncut(W, sc_labels)

        r = {
            'dataset': ds, 'model': model, 'K': K, 'N': N,
            'km_acc': km_res['accuracy'], 'km_nmi': km_res['nmi'],
            'km_rcut': float(km_rcut), 'km_ncut': float(km_ncut),
            'sc_acc': sc_res['accuracy'], 'sc_nmi': sc_res['nmi'],
            'sc_rcut': float(sc_rcut), 'sc_ncut': float(sc_ncut),
        }
        results.append(r)
        print(f'KM={km_res["accuracy"]:.3f} SC={sc_res["accuracy"]:.3f}')

with open('results/baselines_with_cuts.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved {len(results)} results')
PYEOF