#!/usr/bin/env python3
import argparse, struct
import numpy as np
import os
import math
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from PIL import Image
from pathlib import Path
import time


def convert_to_bmp(input_path, output_path=None):
    """
    Konwertuje obraz do formatu BMP
    """
    try:
        img = Image.open(input_path)
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".bmp"

        img.save(output_path, "BMP")
        return output_path

    except Exception as e:
        print(f"Błąd podczas konwersji {input_path}: {e}")
        return ""


def pack_bits(ids, bits_per_id):
    if bits_per_id == 0:
        return b''
    packed = bytearray()
    current_byte = 0
    bits_in_current = 0
    for id_val in ids:
        current_byte = (current_byte << bits_per_id) | (id_val & ((1 << bits_per_id) - 1))
        bits_in_current += bits_per_id
        while bits_in_current >= 8:
            bits_in_current -= 8
            packed.append((current_byte >> bits_in_current) & 0xFF)
            current_byte &= (1 << bits_in_current) - 1
    if bits_in_current > 0:
        current_byte <<= (8 - bits_in_current)
        packed.append(current_byte & 0xFF)

    return bytes(packed)


def cluster_compress(input_path, output_path, chunk_size=6, n_clusters=256):
    with open(input_path, "rb") as f:
        data = f.read()
    print(f"Wczytano {len(data)} bajtów z {input_path}")
    is_bmp = data[:2] == b'BM'

    if is_bmp:
        data_offset = struct.unpack("<I", data[10:14])[0]
        print(f"Offset danych obrazu: {data_offset} bajtów")

        header = data[:data_offset]
        body = data[data_offset:]

        print(f"Plik BMP - nagłówek: {len(header)} bajtów, dane: {len(body)} bajtów")
        file_size_in_header = struct.unpack("<I", data[2:6])[0]
        if file_size_in_header != len(data):
            print(f"Uwaga: Rozmiar pliku w nagłówku ({file_size_in_header}) różni się od rzeczywistego ({len(data)})")
    else:
        header = b''
        body = data
        print(f"Plik nie-BMP - brak nagłówka, dane: {len(body)} bajtów")
    body_np = np.frombuffer(body, dtype=np.uint8)
    original_body_size = len(body_np)
    pad = (-len(body_np)) % chunk_size
    if pad:
        print(f"Dodaję {pad} bajtów paddingu do dopasowania segmentów")
        body_np = np.concatenate([body_np, np.zeros(pad, dtype=np.uint8)])
    chunks = body_np.reshape((-1, chunk_size))
    print(f"Segmenty: {len(chunks)} x {chunk_size} bajtów")
    if args.algorithm == "KMEANS":
        Cluster = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42, batch_size=1000)
        print(Cluster)
        labels = Cluster.fit_predict(chunks)
        centers = np.uint8(np.clip(Cluster.cluster_centers_, 0, 255))
    elif args.algorithm == "DBSCAN":
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
        eps_values = [0.3] # jako tablice na przyszlosc
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

    bits_per_id = max(1, math.ceil(math.log2(len(centers))))
    with open(output_path, "wb") as f:
        # 1. Flaga czy to BMP (1 bajt)
        f.write(struct.pack("B", 1 if is_bmp else 0))

        # 2. Rozmiar nagłówka (4 bajty)
        header_size = len(header)
        f.write(struct.pack("<I", header_size))

        # 3. Oryginalny rozmiar danych (bez paddingu do segmentów) (4 bajty)
        f.write(struct.pack("<I", original_body_size))

        # 4. Rozmiar segmentu/chunk_size (4 bajty)
        f.write(struct.pack("<I", chunk_size))

        # 5. Liczba bitów na ID (1 bajt)
        f.write(struct.pack("B", bits_per_id))

        # 6. Liczba klastrów (4 bajty)
        f.write(struct.pack("<I", len(centers)))

        # 7. Liczba segmentów (4 bajty)
        f.write(struct.pack("<I", len(labels)))

        # 8. Nagłówek (jeśli istnieje) - ZACHOWUJEMY ORYGINALNY NAGŁÓWEK
        if header:
            f.write(header)

        # 9. Centroidy (wszystkie po kolei)
        for center in centers:
            f.write(center.tobytes())

        # 10. Lista ID segmentów (spakowane bity)
        packed_labels = pack_bits(labels, bits_per_id)
        f.write(packed_labels)

    input_size = len(data)
    output_size = os.path.getsize(output_path)
    compression_ratio = output_size / input_size

    print(f"\nKompresja zakończona!")
    print(f"Plik wejściowy: {input_size} bajtów")
    print(f"Plik wyjściowy: {output_size} bajtów")
    print(f"Stopień kompresji: {compression_ratio:.2%}")
    print(f"Oszczędność: {(1 - compression_ratio):.1%}")

    return compression_ratio


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kompresja obrazów do własnego formatu z algorytmem K-means")
    p.add_argument("input", help="Plik wejściowy (obraz)")
    p.add_argument("output", help="Plik wyjściowy (skompresowany)")
    p.add_argument("algorithm", help="Wybierz algorytm klasteryzacji")
    p.add_argument("--auto", help="Czy pytać o parametry", action="store_true")
    args = p.parse_args()

    print(f"Plik wejściowy: {args.input}")
    print(f"Plik wyjściowy: {args.output}")

    bmp_path = convert_to_bmp(args.input)
    if not bmp_path:
        print("Błąd konwersji do BMP!")
        exit(1)

    args.input = bmp_path

    if not args.auto:
        while True:
            try:
                chunk_size_input = input("\nRozmiar segmentu (chunk_size) [domyślnie 6]: ").strip()
                chunk_size = int(chunk_size_input) if chunk_size_input else 6
                if chunk_size <= 0:
                    print("Rozmiar musi być > 0")
                    continue
                break
            except ValueError:
                print("Wprowadź liczbę całkowitą")
    else:
        chunk_size = 6

    if not args.auto:
        while True:
            try:
                n_clusters_input = input("Liczba klastrów (n_clusters) [domyślnie 256]: ").strip()
                n_clusters = int(n_clusters_input) if n_clusters_input else 256
                if n_clusters <= 0:
                    print("Liczba klastrów musi być > 0")
                    continue
                break
            except ValueError:
                print("Wprowadź liczbę całkowitą")
    else:
        n_clusters = 256

    print(f"\nParametry:")
    print(f"Rozmiar segmentu: {chunk_size}")
    print(f"Liczba klastrów: {n_clusters}")
    print()

    cluster_compress(args.input, args.output, chunk_size, n_clusters)