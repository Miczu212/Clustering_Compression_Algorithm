#!/usr/bin/env python3
import argparse, struct
import numpy as np
import math
import os

def unpack_bits(data, bits_per_id, num_ids):
    if not data or bits_per_id == 0:
        return [0] * num_ids if num_ids > 0 else []
    
    ids = []
    current_byte = 0
    bits_available = 0
    byte_idx = 0
    data_len = len(data)
    mask = (1 << bits_per_id) - 1
    
    for _ in range(num_ids):
        while bits_available < bits_per_id:
            if byte_idx >= data_len:
                ids.append(0)
                continue
            current_byte = (current_byte << 8) | data[byte_idx]
            byte_idx += 1
            bits_available += 8
        
        bits_available -= bits_per_id
        id_val = (current_byte >> bits_available) & mask
        current_byte &= (1 << bits_available) - 1
        ids.append(id_val)
    
    return ids

def cluster_decompress(input_path, output_path):
    print(f"Odczytywanie pliku skompresowanego: {input_path}")
    
    with open(input_path, "rb") as f:
        # 1. Flaga czy to BMP
        is_bmp = struct.unpack("B", f.read(1))[0]
        print(f"Flaga BMP: {is_bmp == 1}")
        
        # 2. Rozmiar nagłówka
        header_size = struct.unpack("<I", f.read(4))[0]
        print(f"Rozmiar nagłówka: {header_size} bajtów")
        
        # 3. Oryginalny rozmiar danych (bez paddingu do segmentów)
        original_data_size = struct.unpack("<I", f.read(4))[0]
        print(f"Oryginalny rozmiar danych: {original_data_size} bajtów")
        
        # 4. Rozmiar segmentu/chunk_size
        chunk_size = struct.unpack("<I", f.read(4))[0]
        print(f"Rozmiar segmentu: {chunk_size} bajtów")
        
        # 5. Liczba bitów na ID
        bits_per_id = struct.unpack("B", f.read(1))[0]
        print(f"Bity na ID: {bits_per_id}")
        
        # 6. Liczba klastrów
        n_clusters = struct.unpack("<I", f.read(4))[0]
        print(f"Liczba klastrów: {n_clusters}")
        
        # 7. Liczba segmentów
        num_segments = struct.unpack("<I", f.read(4))[0]
        print(f"Liczba segmentów: {num_segments}")
        
        # 8. Odczyt nagłówka (jeśli istnieje) - ORYGINALNY NAGŁÓWEK
        header = b''
        if header_size > 0:
            header = f.read(header_size)
            if len(header) < header_size:
                raise ValueError("Niekompletny nagłówek")
        
        # 9. Odczyt centroidów
        print(f"Wczytywanie {n_clusters} centroidów...")
        centers = []
        for i in range(n_clusters):
            center_data = f.read(chunk_size)
            if len(center_data) != chunk_size:
                raise ValueError(f"Niekompletny centroid {i}")
            centers.append(np.frombuffer(center_data, dtype=np.uint8))
            
            if (i + 1) % 100 == 0:
                print(f"Wczytano {i+1}/{n_clusters} centroidów...")
        
        print(f"Wczytano wszystkie centroidy")
        
        # 10. Odczyt spakowanych etykiet
        packed_labels = f.read()
        print(f"Rozmiar spakowanych etykiet: {len(packed_labels)} bajtów")


    labels = unpack_bits(packed_labels, bits_per_id, num_segments)
    
    if len(labels) != num_segments:
        if len(labels) < num_segments:
            labels.extend([0] * (num_segments - len(labels)))
        else:
            labels = labels[:num_segments]
    
    print(f"Rozpakowano {len(labels)} etykiet")

    reconstructed_parts = []
    
    for i, label in enumerate(labels):
        if label >= len(centers):
            print(f"Błąd: Etykieta {label} poza zakresem [0, {len(centers)-1}]")
            label = 0
        
        reconstructed_parts.append(centers[label])
        
        if (i + 1) % 10000 == 0:
            print(f"Odtworzono {i+1}/{num_segments} segmentów...")
    
    # Połącz wszystkie segmenty
    reconstructed = np.concatenate(reconstructed_parts)
    print(f"Odtworzono {len(reconstructed)} bajtów danych")
    
    # Przycięcie do oryginalnego rozmiaru (usuń padding dodany w kompresji)
    if len(reconstructed) > original_data_size:
        print(f"Usuwanie paddingu...")
        reconstructed = reconstructed[:original_data_size]
    elif len(reconstructed) < original_data_size:
        padding = np.zeros(original_data_size - len(reconstructed), dtype=np.uint8)
        reconstructed = np.concatenate([reconstructed, padding])

    if is_bmp and header:
        full_data = header + reconstructed.tobytes()

        new_file_size = len(full_data)
        header_bytearray = bytearray(header)
        header_bytearray[2:6] = struct.pack("<I", new_file_size)

        if len(header) >= 54:
            image_size = len(reconstructed)
            header_bytearray[34:38] = struct.pack("<I", image_size)
        
        full_data = bytes(header_bytearray) + reconstructed.tobytes()
    else:
        full_data = reconstructed.tobytes()

    with open(output_path, "wb") as f:
        f.write(full_data)
    
    print(f"\nDekompresja zakończona pomyślnie!")
    print(f"Plik wyjściowy: {output_path}")
    print(f"Rozmiar: {len(full_data)} bajtów")
    if is_bmp and len(full_data) >= 2:
        if full_data[:2] == b'BM':
            try:
                width = struct.unpack("<I", full_data[18:22])[0]
                height = struct.unpack("<I", full_data[22:26])[0]
                bpp = struct.unpack("<H", full_data[28:30])[0]
                print(f"Wymiary: {width}x{height}, {bpp} bpp")
            except:
                pass
        else:
            print(f"Uwaga: Nagłówek BMP może być uszkodzony")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dekompresja pliku skompresowanego z algorytmem K-means")
    p.add_argument("input", help="Plik wejściowy (skompresowany)")
    p.add_argument("output", help="Plik wyjściowy")
    
    args = p.parse_args()
    
    try:
        cluster_decompress(args.input, args.output)
    except FileNotFoundError:
        print(f"Plik {args.input} nie istnieje")
    except Exception as e:
        print(f"Błąd podczas dekompresji: {e}")
