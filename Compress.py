#!/usr/bin/env python3
import argparse, struct
import numpy as np
import os
import math
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from pathlib import Path 

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
        print(f"Dodaję {pad} bajtów paddingu do dopasowania segmentów do chunk_size")
        body_np = np.concatenate([body_np, np.zeros(pad, dtype=np.uint8)])
    
    chunks = body_np.reshape((-1, chunk_size))
    print(f"Segmenty: {len(chunks)} x {chunk_size} bajtów")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42, batch_size=1000)
    labels = kmeans.fit_predict(chunks)
    centers = np.uint8(np.clip(kmeans.cluster_centers_, 0, 255))

    bits_per_id = max(1, math.ceil(math.log2(n_clusters)))
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
        f.write(struct.pack("<I", n_clusters))
        
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
    p.add_argument("--auto", help="Czy pytać o parametry",action="store_true")
    args = p.parse_args()

    print(f"Plik wejściowy: {args.input}")
    print(f"Plik wyjściowy: {args.output}")
    
    # Pobierz parametry
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
        n_clusters=256
    
    print(f"\nParametry:")
    print(f"Rozmiar segmentu: {chunk_size}")
    print(f"Liczba klastrów: {n_clusters}")
    print()
    
    cluster_compress(args.input, args.output, chunk_size, n_clusters)
