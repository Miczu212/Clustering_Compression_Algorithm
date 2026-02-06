#!/usr/bin/env python3
import argparse, struct
import numpy as np
import math


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


def compare_images_pixelwise(original_path, decompressed_path):
    """
    Porównuje dwa obrazy BMP na poziomie pikseli z uwzględnieniem podobieństwa
    """
    print(f"\n{'=' * 60}")
    print("PORÓWNANIE PIKSELI Z ORYGINALNYM OBRAZEM")
    print('=' * 60)

    try:
        with open(original_path, 'rb') as f:
            original_data = f.read()
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku oryginalnego: {original_path}")
        return

    with open(decompressed_path, 'rb') as f:
        decompressed_data = f.read()

    # Sprawdzenie czy to prawdziwe BMP
    if original_data[:2] != b'BM' or decompressed_data[:2] != b'BM':
        print("UWAGA: Jeden z plików nie jest prawidłowym BMP")
        return

    # Odczytanie informacji z nagłówka
    orig_width = struct.unpack("<I", original_data[18:22])[0]
    orig_height = struct.unpack("<I", original_data[22:26])[0]
    orig_bpp = struct.unpack("<H", original_data[28:30])[0]
    orig_offset = struct.unpack("<I", original_data[10:14])[0]
    orig_img_size = struct.unpack("<I", original_data[34:38])[0]

    decomp_width = struct.unpack("<I", decompressed_data[18:22])[0]
    decomp_height = struct.unpack("<I", decompressed_data[22:26])[0]
    decomp_bpp = struct.unpack("<H", decompressed_data[28:30])[0]
    decomp_offset = struct.unpack("<I", decompressed_data[10:14])[0]
    decomp_img_size = struct.unpack("<I", decompressed_data[34:38])[0]

    print(f"ORIGINALNY: {orig_width}x{orig_height}, {orig_bpp} bpp")
    print(f"DEKOMPRESJA: {decomp_width}x{decomp_height}, {decomp_bpp} bpp")

    if orig_width != decomp_width or orig_height != decomp_height or orig_bpp != decomp_bpp:
        print("UWAGA: Obrazy mają różne wymiary lub głębię bitową!")
        print("Porównanie może być niedokładne.")

    # Wybór mniejszych wymiarów do porównania
    width = min(orig_width, decomp_width)
    height = min(orig_height, decomp_height)

    # Odczyt danych obrazu
    orig_img_data = original_data[orig_offset:orig_offset + orig_img_size]
    decomp_img_data = decompressed_data[decomp_offset:decomp_offset + decomp_img_size]

    # Przetwarzanie w zależności od bpp
    if orig_bpp == 24:  # 24-bitowe RGB
        print("\nAnaliza 24-bitowego obrazu RGB")
        return compare_24bit_images(orig_img_data, decomp_img_data, width, height)
    elif orig_bpp == 8:  # 8-bitowe odcienie szarości
        print("\nAnaliza 8-bitowego obrazu w odcieniach szarości")
        return compare_8bit_images(orig_img_data, decomp_img_data, width, height)
    else:
        print(f"\nNieobsługiwana głębia bitowa: {orig_bpp} bpp")
        return


def compare_24bit_images(orig_data, decomp_data, width, height):
    """
    Porównuje dwa 24-bitowe obrazy RGB
    """
    # Oblicz rozmiar wiersza z paddingiem (do wielokrotności 4 bajtów)
    row_size = width * 3
    padding = (4 - (row_size % 4)) % 4
    row_size_with_padding = row_size + padding

    total_pixels = width * height

    # Przygotowanie danych
    orig_pixels = []
    decomp_pixels = []

    # Przetwarzanie wiersz po wierszu
    for row in range(height):
        orig_row_start = row * row_size_with_padding
        decomp_row_start = row * row_size_with_padding

        for col in range(width):
            pixel_start = col * 3

            # Piksel w formacie BGR (BMP przechowuje w kolejności B, G, R)
            orig_b = orig_data[orig_row_start + pixel_start]
            orig_g = orig_data[orig_row_start + pixel_start + 1]
            orig_r = orig_data[orig_row_start + pixel_start + 2]

            decomp_b = decomp_data[decomp_row_start + pixel_start]
            decomp_g = decomp_data[decomp_row_start + pixel_start + 1]
            decomp_r = decomp_data[decomp_row_start + pixel_start + 2]

            orig_pixels.append((orig_r, orig_g, orig_b))  # Konwertujemy do RGB
            decomp_pixels.append((decomp_r, decomp_g, decomp_b))

    # Analiza porównawcza
    identical_pixels = 0
    similar_pixels = 0  # Piksele z małą różnicą
    different_pixels = 0

    max_difference = 0
    total_difference = 0
    mse_total = 0.0  # Mean Squared Error
    psnr_score = 0.0

    pixel_differences = []

    for i in range(total_pixels):
        orig_r, orig_g, orig_b = orig_pixels[i]
        decomp_r, decomp_g, decomp_b = decomp_pixels[i]

        # Różnice dla każdego kanału
        diff_r = abs(int(orig_r) - int(decomp_r))
        diff_g = abs(int(orig_g) - int(decomp_g))
        diff_b = abs(int(orig_b) - int(decomp_b))

        # Średnia różnica dla piksela
        avg_diff = (diff_r + diff_g + diff_b) / 3.0

        # Maksymalna różnica dla tego piksela
        max_pixel_diff = max(diff_r, diff_g, diff_b)
        max_difference = max(max_difference, max_pixel_diff)
        total_difference += avg_diff

        # Obliczanie MSE
        mse_total += (diff_r ** 2 + diff_g ** 2 + diff_b ** 2) / 3.0

        # Klasyfikacja pikseli
        if diff_r == 0 and diff_g == 0 and diff_b == 0:
            identical_pixels += 1
        elif avg_diff <= 10:  # Piksele z małą różnicą (do 10 na kanale)
            similar_pixels += 1
            different_pixels += 1
        else:
            different_pixels += 1

        # Zapisujemy pierwsze 10 różnych pikseli
        if diff_r > 0 or diff_g > 0 or diff_b > 0:
            if len(pixel_differences) < 10:
                row = i // width
                col = i % width
                pixel_differences.append((row, col, orig_r, orig_g, orig_b, decomp_r, decomp_g, decomp_b, avg_diff))

    # Obliczanie statystyk
    avg_difference = total_difference / total_pixels if total_pixels > 0 else 0
    mse = mse_total / total_pixels if total_pixels > 0 else 0

    # Obliczanie PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        psnr_score = 20 * math.log10(255.0 / math.sqrt(mse))
    else:
        psnr_score = float('inf')

    similarity_percent = (identical_pixels / total_pixels * 100) if total_pixels > 0 else 0
    similar_percent = (similar_pixels / total_pixels * 100) if total_pixels > 0 else 0

    print(f"Statystyki porównania pikseli:")
    print(f"  Łączna liczba pikseli:     {total_pixels}")
    print(f"  Identyczne piksele:        {identical_pixels} ({similarity_percent:.2f}%)")
    print(f"  Podobne piksele (Δ≤10):    {similar_pixels} ({similar_percent:.2f}%)")
    print(f"  Różne piksele:             {different_pixels} ({(100 - similarity_percent):.2f}%)")
    print(f"  Średnia różnica na piksel: {avg_difference:.2f}")
    print(f"  Maksymalna różnica:        {max_difference}")
    print(f"  MSE (Mean Squared Error):  {mse:.2f}")
    print(f"  PSNR:                      {psnr_score:.2f} dB")

    # Interpretacja PSNR
    if psnr_score > 40:
        print("  Jakość:                   Bardzo dobra")
    elif psnr_score > 30:
        print("  Jakość:                   Dobra")
    elif psnr_score > 20:
        print("  Jakość:                   Umiarkowana")
    else:
        print("  Jakość:                   Słaba")

    # Wyświetlenie przykładowych różnic
    if pixel_differences:
        print(f"\nPrzykładowe różne piksele (pierwsze {len(pixel_differences)}):")
        print("  Wiersz | Kolumna | Oryginalny (R,G,B) | Dekompresja (R,G,B) | Różnica")
        print("  " + "-" * 65)
        for row, col, orig_r, orig_g, orig_b, decomp_r, decomp_g, decomp_b, avg_diff in pixel_differences:
            print(f"  {row:6d} | {col:6d} | ({orig_r:3d},{orig_g:3d},{orig_b:3d}) | "
                  f"({decomp_r:3d},{decomp_g:3d},{decomp_b:3d}) | {avg_diff:.1f}")

    # Analiza histogramu różnic
    print(f"\nHistogram różnic pikseli:")
    diff_bins = [0] * 10  # 0, 1-10, 11-20, ..., 81-90, 91-255
    for i in range(total_pixels):
        orig_r, orig_g, orig_b = orig_pixels[i]
        decomp_r, decomp_g, decomp_b = decomp_pixels[i]
        avg_diff = (abs(orig_r - decomp_r) + abs(orig_g - decomp_g) + abs(orig_b - decomp_b)) / 3.0

        if avg_diff == 0:
            diff_bins[0] += 1
        else:
            bin_idx = min(9, int(avg_diff / 25.6))  # Dzielimy zakres 0-255 na 10 przedziałów
            diff_bins[bin_idx] += 1

    print("  Przedział różnicy | Liczba pikseli | Procent")
    print("  " + "-" * 45)
    bin_labels = ["0 (identyczne)", "1-25", "26-51", "52-76", "77-102",
                  "103-127", "128-153", "154-178", "179-204", "205-255"]
    for i in range(10):
        percent = (diff_bins[i] / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {bin_labels[i]:15s} | {diff_bins[i]:13d} | {percent:6.2f}%")


def compare_8bit_images(orig_data, decomp_data, width, height):
    """
    Porównuje dwa 8-bitowe obrazy w odcieniach szarości
    """
    # Oblicz rozmiar wiersza z paddingiem
    row_size = width
    padding = (4 - (row_size % 4)) % 4
    row_size_with_padding = row_size + padding

    total_pixels = width * height

    # Przygotowanie danych
    orig_pixels = []
    decomp_pixels = []

    # Przetwarzanie wiersz po wierszu
    for row in range(height):
        orig_row_start = row * row_size_with_padding
        decomp_row_start = row * row_size_with_padding

        for col in range(width):
            orig_pixel = orig_data[orig_row_start + col]
            decomp_pixel = decomp_data[decomp_row_start + col]

            orig_pixels.append(orig_pixel)
            decomp_pixels.append(decomp_pixel)

    # Analiza porównawcza
    identical_pixels = 0
    similar_pixels = 0  # Piksele z małą różnicą
    different_pixels = 0

    max_difference = 0
    total_difference = 0
    mse_total = 0.0

    pixel_differences = []

    for i in range(total_pixels):
        orig_val = orig_pixels[i]
        decomp_val = decomp_pixels[i]

        diff = abs(int(orig_val) - int(decomp_val))
        max_difference = max(max_difference, diff)
        total_difference += diff
        mse_total += diff ** 2

        # Klasyfikacja pikseli
        if diff == 0:
            identical_pixels += 1
        elif diff <= 10:  # Piksele z małą różnicą
            similar_pixels += 1
            different_pixels += 1
        else:
            different_pixels += 1

        # Zapisujemy pierwsze 10 różnych pikseli
        if diff > 0 and len(pixel_differences) < 10:
            row = i // width
            col = i % width
            pixel_differences.append((row, col, orig_val, decomp_val, diff))

    # Obliczanie statystyk
    avg_difference = total_difference / total_pixels if total_pixels > 0 else 0
    mse = mse_total / total_pixels if total_pixels > 0 else 0

    # Obliczanie PSNR
    if mse > 0:
        psnr_score = 20 * math.log10(255.0 / math.sqrt(mse))
    else:
        psnr_score = float('inf')

    similarity_percent = (identical_pixels / total_pixels * 100) if total_pixels > 0 else 0
    similar_percent = (similar_pixels / total_pixels * 100) if total_pixels > 0 else 0

    print(f"Statystyki porównania pikseli (odcienie szarości):")
    print(f"  Łączna liczba pikseli:     {total_pixels}")
    print(f"  Identyczne piksele:        {identical_pixels} ({similarity_percent:.2f}%)")
    print(f"  Podobne piksele (Δ≤10):    {similar_pixels} ({similar_percent:.2f}%)")
    print(f"  Różne piksele:             {different_pixels} ({(100 - similarity_percent):.2f}%)")
    print(f"  Średnia różnica na piksel: {avg_difference:.2f}")
    print(f"  Maksymalna różnica:        {max_difference}")
    print(f"  MSE (Mean Squared Error):  {mse:.2f}")
    print(f"  PSNR:                      {psnr_score:.2f} dB")

    # Interpretacja PSNR
    if psnr_score > 40:
        print("  Jakość:                   Bardzo dobra")
    elif psnr_score > 30:
        print("  Jakość:                   Dobra")
    elif psnr_score > 20:
        print("  Jakość:                   Umiarkowana")
    else:
        print("  Jakość:                   Słaba")

    # Wyświetlenie przykładowych różnic
    if pixel_differences:
        print(f"\nPrzykładowe różne piksele (pierwsze {len(pixel_differences)}):")
        print("  Wiersz | Kolumna | Oryginalny | Dekompresja | Różnica")
        print("  " + "-" * 55)
        for row, col, orig_val, decomp_val, diff in pixel_differences:
            print(f"  {row:6d} | {col:6d} | {orig_val:9d} | {decomp_val:11d} | {diff:7d}")

    # Analiza histogramu różnic
    print(f"\nHistogram różnic pikseli:")
    diff_bins = [0] * 10
    for i in range(total_pixels):
        diff = abs(int(orig_pixels[i]) - int(decomp_pixels[i]))

        if diff == 0:
            diff_bins[0] += 1
        else:
            bin_idx = min(9, int(diff / 25.6))
            diff_bins[bin_idx] += 1

    print("  Przedział różnicy | Liczba pikseli | Procent")
    print("  " + "-" * 45)
    bin_labels = ["0 (identyczne)", "1-25", "26-51", "52-76", "77-102",
                  "103-127", "128-153", "154-178", "179-204", "205-255"]
    for i in range(10):
        percent = (diff_bins[i] / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {bin_labels[i]:15s} | {diff_bins[i]:13d} | {percent:6.2f}%")


def compare_files_bitwise(original_path, decompressed_path, is_bmp=False):
    print("PORÓWNANIE BITOWE Z ORYGINALNYM PLIKIEM")
    try:
        with open(original_path, 'rb') as f:
            original_data = f.read()
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku oryginalnego: {original_path}")
        return

    with open(decompressed_path, 'rb') as f:
        decompressed_data = f.read()

    def bytes_to_bit_array(data):
        bit_array = []
        for byte in data:
            for i in range(7, -1, -1):
                bit = (byte >> i) & 1
                bit_array.append(bit)
        return bit_array

    orig_bits = bytes_to_bit_array(original_data)
    decomp_bits = bytes_to_bit_array(decompressed_data)
    orig_bit_size = len(orig_bits)
    decomp_bit_size = len(decomp_bits)

    print(f"Rozmiar pliku oryginalnego:   {orig_bit_size} bitów")
    print(f"Rozmiar pliku zdekompresowanego: {decomp_bit_size} bitów")
    if is_bmp and len(original_data) >= 54 and len(decompressed_data) >= 54:

        print("\nSpecyficzne porównanie dla BMP:")
        orig_offset = struct.unpack("<I", original_data[10:14])[0]
        decomp_offset = struct.unpack("<I", decompressed_data[10:14])[0]
        print(f"Offset danych obrazu (oryginalny):   {orig_offset} bajtów ({orig_offset * 8} bitów)")
        print(f"Offset danych obrazu (zdekompresowany): {decomp_offset} bajtów ({decomp_offset * 8} bitów)")
        orig_img_size = struct.unpack("<I", original_data[34:38])[0] if len(original_data) >= 38 else 0
        decomp_img_size = struct.unpack("<I", decompressed_data[34:38])[0] if len(decompressed_data) >= 38 else 0
        print(f"Rozmiar danych obrazu (oryginalny): ({orig_img_size * 8} bitów)")
        print(f"Rozmiar danych obrazu (zdekompresowany): ({decomp_img_size * 8} bitów)")
        orig_img_start_bit = orig_offset * 8
        orig_img_end_bit = orig_img_start_bit + orig_img_size * 8
        decomp_img_start_bit = decomp_offset * 8
        decomp_img_end_bit = decomp_img_start_bit + decomp_img_size * 8
        orig_img_bits = orig_bits[orig_img_start_bit:orig_img_end_bit]
        decomp_img_bits = decomp_bits[decomp_img_start_bit:decomp_img_end_bit]
        min_img_bit_size = min(len(orig_img_bits), len(decomp_img_bits))
        max_img_bit_size = max(len(orig_img_bits), len(decomp_img_bits))

        if min_img_bit_size == 0:
            return
        different_bits = 0
        same_bits = 0
        bit_differences = []
        for i in range(min_img_bit_size):
            if orig_img_bits[i] != decomp_img_bits[i]:
                different_bits += 1
                if len(bit_differences) < 20:
                    bit_differences.append((i, orig_img_bits[i], decomp_img_bits[i]))
            else:
                same_bits += 1
        if max_img_bit_size > min_img_bit_size:
            different_bits += (max_img_bit_size - min_img_bit_size)
        similarity = (same_bits / max_img_bit_size * 100) if max_img_bit_size > 0 else 0
        print(f"\nPorównanie BITOWE DANYCH OBRAZU (bez nagłówka):")
        print(f"  Bitów identycznych:   {same_bits}")
        print(f"  Bitów różnych:        {different_bits}")
        print(f"  Podobieństwo:          {similarity:.2f}%")
        print(f"  Procent bitów poprawnych: {(same_bits / max_img_bit_size * 100):.4f}%")
        print(f"  Współczynnik błędów bitowych (BER): {(different_bits / max_img_bit_size * 100):.4f}%")
        header_bit_size = min(orig_offset, decomp_offset, 54) * 8
        print(f"\nPorównanie BITOWE NAGŁÓWKÓW (pierwszych {header_bit_size} bitów):")
        header_diff_bits = 0
        for i in range(header_bit_size):
            if orig_bits[i] != decomp_bits[i]:
                header_diff_bits += 1
        print(f"  Różnych bitów w nagłówku: {header_diff_bits}/{header_bit_size}")
        print(
            f"  Procent poprawnych bitów w nagłówku: {((header_bit_size - header_diff_bits) / header_bit_size * 100):.2f}%")


    else:
        min_bit_size = min(orig_bit_size, decomp_bit_size)
        max_bit_size = max(orig_bit_size, decomp_bit_size)
        different_bits = 0
        same_bits = 0
        for i in range(min_bit_size):

            if orig_bits[i] != decomp_bits[i]:
                different_bits += 1
            else:
                same_bits += 1
        if max_bit_size > min_bit_size:
            different_bits += (max_bit_size - min_bit_size)
        similarity = (same_bits / max_bit_size * 100) if max_bit_size > 0 else 0
        print(f"\nPorównanie BITOWE CAŁYCH PLIKÓW:")
        print(f"  Bitów identycznych:   {same_bits}")
        print(f"  Bitów różnych:        {different_bits}")
        print(f"  Podobieństwo:          {similarity:.2f}%")
        print(f"  Bit Error Rate (BER): {(different_bits / max_bit_size * 100):.4f}%")


def cluster_decompress(input_path, output_path, original_path=None):
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
                print(f"Wczytano {i + 1}/{n_clusters} centroidów...")

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
            print(f"Błąd: Etykieta {label} poza zakresem [0, {len(centers) - 1}]")
            label = 0

        reconstructed_parts.append(centers[label])

        if (i + 1) % 10000 == 0:
            print(f"Odtworzono {i + 1}/{num_segments} segmentów...")

    reconstructed = np.concatenate(reconstructed_parts)
    print(f"Odtworzono {len(reconstructed)} bajtów danych")
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
    if original_path:
        if is_bmp:
            # Dla obrazów używamy porównania pikseli
            compare_images_pixelwise(original_path, output_path)
        else:
            # Dla innych plików używamy porównania bitowego
            compare_files_bitwise(original_path, output_path, is_bmp=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dekompresja pliku skompresowanego z algorytmem K-means")
    p.add_argument("input", help="Plik wejściowy (skompresowany)")
    p.add_argument("output", help="Plik wyjściowy")
    p.add_argument("--check-original", "-c", help="Ścieżka do oryginalnego pliku do porównania po dekompresji")

    args = p.parse_args()

    try:
        cluster_decompress(args.input, args.output, args.check_original)
    except FileNotFoundError as e:
        print(f"Błąd: Nie znaleziono pliku: {e.filename}")
    except Exception as e:
        print(f"Błąd podczas dekompresji: {e}")