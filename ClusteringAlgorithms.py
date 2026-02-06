import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans, DBSCAN
def DBSCANCluster(chunks,chunk_size,n_clusters):
    print("\n=== DBSCAN z ograniczeniem czasu ===")
    print(f"Liczba punktów: {len(chunks)}")
    max_samples_for_dbscan = 50000
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
        n_forced = min(n_clusters, 32)
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