import math
import random
import argparse
import threading
import queue
from collections import Counter


def parse_size(size_str):
    size_str = size_str.strip().upper()
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'K': 1024,
        'M': 1024 * 1024,
        'G': 1024 * 1024 * 1024
    }
    unit = ''
    value_str = size_str
    for unit_key in sorted(multipliers.keys(), key=len, reverse=True):
        if size_str.endswith(unit_key):
            unit = unit_key
            value_str = size_str[:-len(unit_key)]
            break
    if not unit:
        unit = 'B'
    try:
        value = float(value_str)
    except ValueError:
        raise ValueError(f"Nieprawidłowa wartość rozmiaru: '{value_str}' w '{size_str}'")
    bytes_count = int(value * multipliers[unit])
    return bytes_count


def calculate_entropy(data):
    if not data:
        return 0
    byte_counts = Counter(data)
    total_bytes = len(data)
    entropy = 0
    for count in byte_counts.values():
        probability = count / total_bytes
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def generate_initial_data(target_entropy, size_bytes):
    if target_entropy < 0.5:
        byte_val = random.randint(0, 255)
        if random.random() < 0.1:
            byte2 = random.randint(0, 255)
            while byte2 == byte_val:
                byte2 = random.randint(0, 255)
            data = []
            for i in range(size_bytes):
                if random.random() < 0.9:
                    data.append(byte_val)
                else:
                    data.append(byte2)
            return bytes(data)
        else:
            return bytes([byte_val] * size_bytes)
    elif target_entropy < 2.0:
        num_unique = random.randint(2, 3)
        unique_bytes = random.sample(range(256), num_unique)
        weights = [random.uniform(0.7, 1.0) for _ in range(num_unique)]
        if target_entropy < 1.0:
            weights[0] = weights[0] * 3
        total = sum(weights)
        weights = [w / total for w in weights]
        data = []
        for _ in range(size_bytes):
            r = random.random()
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    data.append(unique_bytes[i])
                    break
        return bytes(data)
    elif target_entropy < 4.0:
        max_val_from_entropy = int(2 ** (target_entropy / 1.5))
        max_val = min(10, max_val_from_entropy)
        min_val = 4
        if max_val < min_val:
            max_val = min_val
        num_unique = random.randint(min_val, max_val)
        unique_bytes = random.sample(range(256), num_unique)
        weights = [random.uniform(0.5, 1.5) for _ in range(num_unique)]
        if target_entropy < 3.0:
            weights.sort(reverse=True)
        total = sum(weights)
        weights = [w / total for w in weights]
        data = []
        for _ in range(size_bytes):
            r = random.random()
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    data.append(unique_bytes[i])
                    break
        return bytes(data)
    elif target_entropy < 6.0:
        max_val_from_entropy = int(2 ** (target_entropy / 1.2))
        max_val = min(50, max_val_from_entropy)
        min_val = 10
        if max_val < min_val:
            max_val = min_val
        num_unique = random.randint(min_val, max_val)
        unique_bytes = random.sample(range(256), num_unique)
        weights = [random.uniform(0.8, 1.2) for _ in range(num_unique)]
        total = sum(weights)
        weights = [w / total for w in weights]
        data = []
        for _ in range(size_bytes):
            r = random.random()
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    data.append(unique_bytes[i])
                    break
        return bytes(data)
    elif target_entropy < 7.5:
        num_unique = random.randint(100, 200)
        unique_bytes = random.sample(range(256), num_unique)
        weights = [random.uniform(0.9, 1.1) for _ in range(num_unique)]
        total = sum(weights)
        weights = [w / total for w in weights]
        data = []
        for _ in range(size_bytes):
            r = random.random()
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    data.append(unique_bytes[i])
                    break
        return bytes(data)
    else:
        return random.randbytes(size_bytes)


def adjust_data_to_target_entropy(data, target_entropy):
    current_data = list(data)
    current_entropy = calculate_entropy(data)
    while True:
        if abs(current_entropy - target_entropy) <= 0.3:
            return bytes(current_data), current_entropy
        byte_counts = Counter(current_data)
        entropy_diff = target_entropy - current_entropy
        if entropy_diff > 0:
            most_common_byte, most_common_count = byte_counts.most_common(1)[0]
            changes_needed = int(most_common_count * abs(entropy_diff) * 0.3)
            changes_needed = max(1, min(changes_needed, most_common_count // 2))
            positions = [i for i, b in enumerate(current_data) if b == most_common_byte]
            if len(positions) > 0:
                positions_to_change = random.sample(positions, min(changes_needed, len(positions)))
                for pos in positions_to_change:
                    available_bytes = [b for b in range(256) if b != most_common_byte]
                    new_byte = random.choice(available_bytes)
                    current_data[pos] = new_byte
        else:
            least_common = byte_counts.most_common()[-5:]
            most_common_byte = byte_counts.most_common(1)[0][0]
            total_rare_count = sum(count for _, count in least_common)
            changes_needed = int(total_rare_count * abs(entropy_diff) * 0.3)
            changes_needed = max(1, changes_needed)
            rare_positions = []
            for rare_byte, _ in least_common:
                positions = [i for i, b in enumerate(current_data) if b == rare_byte]
                rare_positions.extend(positions)
            if len(rare_positions) > 0:
                positions_to_change = random.sample(
                    rare_positions,
                    min(changes_needed, len(rare_positions))
                )
                for pos in positions_to_change:
                    current_data[pos] = most_common_byte
        current_entropy = calculate_entropy(bytes(current_data))


def worker_generate_data(worker_id, target_entropy, size_bytes, result_queue, stop_event):
    attempts = 0
    best_data = None
    best_entropy_diff = float('inf')
    best_entropy = 0
    while not stop_event.is_set():
        attempts += 1
        data = generate_initial_data(target_entropy, size_bytes)
        adjusted_data, actual_entropy = adjust_data_to_target_entropy(data, target_entropy)
        entropy_diff = abs(actual_entropy - target_entropy)
        if entropy_diff < best_entropy_diff:
            best_entropy_diff = entropy_diff
            best_data = adjusted_data
            best_entropy = actual_entropy
        if entropy_diff <= 0.3:
            result_queue.put({
                'worker_id': worker_id,
                'data': adjusted_data,
                'entropy': actual_entropy,
                'attempts': attempts,
                'in_tolerance': True
            })
            return
    if best_data is not None:
        result_queue.put({
            'worker_id': worker_id,
            'data': best_data,
            'entropy': best_entropy,
            'attempts': attempts,
            'in_tolerance': False
        })


def multi_threaded_generate_data(target_entropy, size_bytes, num_threads=4):
    result_queue = queue.Queue()
    stop_event = threading.Event()
    threads = []
    actual_threads = min(num_threads, max(1, int(size_bytes / 100000)))
    for i in range(actual_threads):
        thread = threading.Thread(
            target=worker_generate_data,
            args=(i, target_entropy, size_bytes, result_queue, stop_event),
            daemon=True
        )
        threads.append(thread)
        thread.start()
    results = []
    while True:
        result = result_queue.get()
        results.append(result)
        if result['in_tolerance']:
            stop_event.set()
            break
    stop_event.set()
    for thread in threads:
        thread.join()
    if results:
        in_tolerance_results = [r for r in results if r['in_tolerance']]
        if in_tolerance_results:
            best_result = in_tolerance_results[0]
        else:
            best_result = min(results, key=lambda r: abs(r['entropy'] - target_entropy))
        return best_result['data'], best_result['entropy'], len(results)
    data = generate_initial_data(target_entropy, size_bytes)
    adjusted_data, actual_entropy = adjust_data_to_target_entropy(data, target_entropy)
    return adjusted_data, actual_entropy, 0


def format_size(size_bytes):
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} bajtów"


def main():
    parser = argparse.ArgumentParser(
        description='Generuj dane binarne o określonej entropii obliczanej w grupach do bajta (wielowątkowo)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--entropy', type=float, required=True,
                        help='Docelowa entropia (0-8)')
    parser.add_argument('--size', type=str, required=True,
                        help='Rozmiar danych')
    parser.add_argument('--output', type=str, required=True,
                        help='Nazwa pliku wyjściowego')
    parser.add_argument('--threads', type=int, default=6,
                        help='Liczba wątków do użycia')
    parser.add_argument('--verify', action='store_true',
                        help='Zweryfikuj entropię wygenerowanych danych', default=True)
    args = parser.parse_args()
    if not 0 <= args.entropy <= 8:
        print("Błąd: Entropia musi być w zakresie 0-8")
        return
    if args.threads < 1:
        print("Błąd: Liczba wątków musi być co najmniej 1")
        return
    try:
        size_bytes = parse_size(args.size)
    except ValueError as e:
        print(f"Błąd parsowania rozmiaru: {e}")
        return
    print(f"Generowanie danych o entropii: ~{args.entropy}")
    print(f"Rozmiar: {format_size(size_bytes)} ({size_bytes:,} bajtów)")
    print(f"Wybrano: {args.threads} wątków")
    data, actual_entropy, num_results = multi_threaded_generate_data(
        args.entropy, size_bytes, args.threads
    )
    with open(args.output, 'wb') as f:
        f.write(data)
    print(f"Dane zapisane do {args.output}")
    if args.verify:
        print("\nWeryfikacja:")
        entropy = calculate_entropy(data)
        print(f"Entropia Shannona: {entropy:.4f} bitów/bajt")
        print(f"\nCel: {args.entropy:.4f} bitów/bajt (+/-0.3)")
        print(f"Osiągnięto: {entropy:.4f} bitów/bajt")
        print(f"Różnica: {abs(entropy - args.entropy):.4f}")

if __name__ == "__main__":
    main()