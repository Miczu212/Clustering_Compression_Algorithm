import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
def DBSCANCluster(chunks,chunk_size,max_samples_for_dbscan):
    print("\n=== DBSCAN z ograniczeniem czasu ===")
    print(f"Liczba punktów: {len(chunks)}")
    if len(chunks) > max_samples_for_dbscan:
        print(f"Próbkuję dane do {max_samples_for_dbscan} punktów dla DBSCAN...")
        indices = np.random.choice(len(chunks), max_samples_for_dbscan, replace=False)
        sample_chunks = chunks[indices]
    else:
        sample_chunks = chunks
        indices = np.arange(len(chunks))
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(3, chunk_size))
    chunks_reduced = pca.fit_transform(sample_chunks.astype(np.float32))
    chunks_normalized = (chunks_reduced - chunks_reduced.mean(axis=0)) / (chunks_reduced.std(axis=0) + 1e-8)
    eps_values = [0.3]  # jako tablice na przyszlosc
    min_samples_values = [4]
    best_labels = None
    best_centers = []
    best_valid_labels = []
    best_n_clusters = 0
    start_time = time.time()
    timeout = 30
    for eps in eps_values:
        for min_samples in min_samples_values:
            if time.time() - start_time > timeout:
                print("Timeout! Przerywam szukanie parametrów.")
                break
            print(f"  Testowane eps={eps}, min_samples={min_samples}")
            try:
                Cluster = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="euclidean",
                    algorithm='ball_tree',
                    n_jobs=-1
                )
                labels_test = Cluster.fit_predict(chunks_normalized)
                unique_labels = np.unique(labels_test)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                if n_clusters_found > 0:
                    print(f"    -> Znaleziono {n_clusters_found} klastrów!")
                    centers_test = []
                    valid_labels = []
                    for label in unique_labels:
                        if label != -1:
                            sample_mask = labels_test == label
                            original_indices = indices[sample_mask]
                            cluster_points = chunks[original_indices]
                            if len(cluster_points) > 0:
                                center = np.mean(cluster_points, axis=0)
                                centers_test.append(center)
                                valid_labels.append(label)
                    if n_clusters_found > best_n_clusters:
                        best_n_clusters = n_clusters_found
                        best_labels = labels_test.copy()
                        best_centers = centers_test.copy()
                        best_valid_labels = valid_labels.copy()
            except Exception as e:
                print(f"    Błąd: {e}")
                continue
        if time.time() - start_time > timeout:
            break
    if best_labels is None or len(best_centers) == 0:
        n_forced = min(99999, 32)
        segment_sums = np.sum(chunks, axis=1)
        kmeans_simple = MiniBatchKMeans(n_clusters=n_forced, batch_size=1000, n_init=1)
        sum_labels = kmeans_simple.fit_predict(segment_sums.reshape(-1, 1))
        best_labels = sum_labels
        best_centers = []
        best_valid_labels = list(range(n_forced))
        for i in range(n_forced):
            mask = sum_labels == i
            if np.any(mask):
                center = np.mean(chunks[mask], axis=0)
                best_centers.append(center)
        if len(best_centers) == 0:
            print("Tworzę losowe centroidy...")
            for i in range(n_forced):
                center = np.random.randint(0, 256, size=chunk_size)
                best_centers.append(center)
    else:
        from sklearn.metrics import pairwise_distances_argmin_min

        full_labels = np.zeros(len(chunks), dtype=int)
        for i, label in enumerate(best_labels):
            if label != -1:
                center_idx = np.where(np.array(best_valid_labels) == label)[0]
                if len(center_idx) > 0:
                    full_labels[indices[i]] = center_idx[0]
        remaining_mask = np.ones(len(chunks), dtype=bool)
        remaining_mask[indices] = False
        if np.any(remaining_mask) and len(best_centers) > 0:
            remaining_points = chunks[remaining_mask]
            closest_centers = pairwise_distances_argmin_min(remaining_points, np.array(best_centers))[0]
            full_labels[remaining_mask] = closest_centers
        best_labels = full_labels
    labels = best_labels
    centers = np.array(best_centers)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 0:
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        labels = np.array([label_map[l] for l in labels])
        centers = centers[sorted(unique_labels)]
    centers = np.uint8(np.clip(centers, 0, 255))
    print(f"\nDBSCAN zakończony w {time.time() - start_time:.1f}s")
    print(f"Ostateczna liczba klastrów: {len(centers)}")
    print(f"Wybrano konfigurację z {best_n_clusters} klastrami (najwięcej ze znalezionych)")
    return centers,labels
def KMEANSCluster(chunks,n_clusters):
    Cluster = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42, batch_size=1000)
    labels = Cluster.fit_predict(chunks)
    centers = np.uint8(np.clip(Cluster.cluster_centers_, 0, 255))
    return centers, labels


def GAUSSIANCluster(chunks, chunk_size, n_clusters, max_samples):
    n_samples = len(chunks)
    print(f"Liczba punktów: {n_samples}, docelowe klastry: {n_clusters}")
    use_pca = chunk_size > 10
    if use_pca:
        pca_dim = min(10, chunk_size, n_samples - 1)
        pca = PCA(n_components=pca_dim)
        data_for_gmm = pca.fit_transform(chunks.astype(np.float32))
    else:
        data_for_gmm = chunks.astype(np.float32)
        pca_dim = chunk_size
    data_mean = data_for_gmm.mean(axis=0)
    data_std = data_for_gmm.std(axis=0) + 1e-8
    data_normalized = (data_for_gmm - data_mean) / data_std
    d = data_normalized.shape[1]
    if len(data_normalized) > max_samples:
        indices = np.random.choice(len(data_normalized), max_samples, replace=False)
        sampled_data = data_normalized[indices]
    else:
        sampled_data = data_normalized
        indices = np.arange(len(data_normalized))
    n_train = len(sampled_data)
    print(f"Liczba próbek treningowych: {n_train}, wymiar: {d}")
    strategies = [
        ('full', d * (d + 1) // 2 + 1, 1e-6, 3),
        ('full', d * (d + 1) // 2 + 1, 1e-2, 3),
        ('diag', d + 1, 1e-6, 2),
        ('diag', d + 1, 1e-2, 2),
        ('spherical', 2, 1e-6, 1),
        ('spherical', 2, 1e-2, 1),
    ]
    target_n = min(n_clusters, n_train - 1) if n_train > 1 else 1
    successful_results = []
    for cov_type, min_samples_per_comp, reg, complexity in strategies:
        max_possible_comp = n_train // min_samples_per_comp
        if max_possible_comp < 1:
            continue
        n_try = min(target_n, max_possible_comp)
        for n_comp in range(n_try, 0, -1):
            try:
                print(f"  Próba: {cov_type}, n_comp={n_comp}, reg_covar={reg}")
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=cov_type,
                    reg_covar=reg,
                    random_state=42,
                    n_init=3,
                    max_iter=300,
                    tol=1e-3
                )
                gmm.fit(sampled_data)
                if len(sampled_data) == len(data_normalized):
                    labels_norm = gmm.predict(data_normalized)
                else:
                    labels_sampled = gmm.predict(sampled_data)
                    centers_norm = gmm.means_
                    remaining = np.ones(len(data_normalized), dtype=bool)
                    remaining[indices] = False
                    full_labels = -np.ones(len(data_normalized), dtype=int)
                    full_labels[indices] = labels_sampled
                    if np.any(remaining):
                        remaining_data = data_normalized[remaining]
                        closest = pairwise_distances_argmin_min(remaining_data, centers_norm)[0]
                        full_labels[remaining] = closest
                    labels_norm = full_labels
                if use_pca:
                    centers_norm = gmm.means_
                    centers_denorm = centers_norm * data_std + data_mean
                    centers_orig = pca.inverse_transform(centers_denorm)
                else:
                    centers_norm = gmm.means_
                    centers_orig = centers_norm * data_std + data_mean
                centers = np.uint8(np.clip(centers_orig, 0, 255))
                labels = labels_norm.astype(np.int32)
                unique = np.unique(labels)
                if len(unique) != n_comp:
                    mapping = {old: new for new, old in enumerate(sorted(unique))}
                    labels = np.array([mapping[l] for l in labels])
                    centers = centers[sorted(unique)]

                print(f"  ✓ Sukces! Klastry: {len(centers)}, złożoność: {cov_type}")
                successful_results.append((len(centers), complexity, centers, labels))
                break
            except Exception as e:
                print(f"    Błąd: {e.__class__.__name__}: {str(e)}")
                continue

    if not successful_results:
        return [], []
    successful_results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = successful_results[0]
    print(f"\nWybrano najlepszy wynik: {best[0]} klastrów, złożoność={best[1]}")
    centers, labels = best[2], best[3]
    return centers, labels