from collections import defaultdict, Counter

OMIT = frozenset('qx')

def build_alphabet(text):
    seen = set(text)
    seen.discard('')
    return sorted(seen)

def normalize(raw, omit=OMIT):
    out = []
    for ch in raw:
        if ch in (' ', '\n'):
            out.append(' ')
        elif ch.isalpha():
            low = ch.lower()
            if low not in omit:
                out.append(low)
    return out

def build_ngram_counts(stream, max_order):
    counts = {}
    for order in range(1, max_order + 1):
        counts[order] = defaultdict(lambda: defaultdict(int))
    for i in range(len(stream) - 1):
        nxt = stream[i + 1]
        for order in range(1, max_order + 1):
            start = i - order + 1
            if start < 0:
                continue
            ctx = tuple(stream[start:i + 1])
            counts[order][ctx][nxt] += 1
    return counts

def build_global_freq(stream, alphabet):
    freq = defaultdict(int)
    for ch in stream:
        if ch in alphabet:
            freq[ch] += 1
    total = sum(freq.values()) or 1
    return sorted(alphabet, key=lambda c: -freq[c] / total)

def make_oracle(counts, global_order, max_order):
    def reorder(context):
        for order in range(min(max_order, len(context)), 0, -1):
            ctx_key = tuple(context[-order:])
            if ctx_key in counts[order]:
                followers = counts[order][ctx_key]
                ranked = sorted(followers.keys(), key=lambda c: -followers[c])
                ranked_set = set(ranked)
                tail = [c for c in global_order if c not in ranked_set]
                return ranked + tail
        return global_order
    return reorder

def simulate(stream, alphabet, reorder_fn, max_order):
    total_cost = 0
    total_chars = 0
    for i in range(len(stream)):
        ch = stream[i]
        if ch not in alphabet:
            continue
        start = max(0, i - max_order)
        context = stream[start:i]
        ordering = reorder_fn(context)
        try:
            pos = ordering.index(ch)
        except ValueError:
            pos = len(ordering)
        total_cost += pos
        total_chars += 1
    avg = total_cost / total_chars if total_chars else 0
    return total_cost, total_chars, avg

def static_reorder(global_order):
    def reorder(context):
        return global_order
    return reorder

def build_oracle_table(stream, oracle_fn, order):
    table = {}
    for i in range(len(stream)):
        start = max(0, i - order)
        ctx = tuple(stream[start:i])
        if ctx not in table:
            table[ctx] = tuple(oracle_fn(ctx))
    return table

def kendall_distance(a, b):
    pos_b = {c: i for i, c in enumerate(b)}
    inv = 0
    n = len(a)
    for i in range(n):
        ai = a[i]
        for j in range(i + 1, n):
            aj = a[j]
            if pos_b[ai] > pos_b[aj]:
                inv += 1
    return inv

def analyze_oracle(stream, oracle_fn, global_order, order):
    table = build_oracle_table(stream, oracle_fn, order)
    print(f"=== Oracle analysis (order={order}) ===")
    print(f"Contexts observed: {len(table)}")
    unique_orderings = Counter(table.values())
    print(f"Unique orderings: {len(unique_orderings)}")
    most_common_ordering, count = unique_orderings.most_common(1)[0]
    pct = 100.0 * count / len(table)
    print(f"Most common ordering: {pct:.2f}% of contexts")
    top1 = Counter()
    top3 = Counter()
    top5 = Counter()
    for ordering in table.values():
        top1[ordering[:1]] += 1
        top3[ordering[:3]] += 1
        top5[ordering[:5]] += 1
    print(f"Distinct top-1 prefixes: {len(top1)}")
    print(f"Distinct top-3 prefixes: {len(top3)}")
    print(f"Distinct top-5 prefixes: {len(top5)}")
    global_pos = {c: i for i, c in enumerate(global_order)}
    displacement_sum = 0
    displacement_count = 0
    for ordering in table.values():
        for pos, ch in enumerate(ordering):
            if ch not in global_pos:
                continue
            displacement_sum += abs(pos - global_pos[ch])
            displacement_count += 1
    avg_displacement = displacement_sum / displacement_count if displacement_count else 0
    print(f"Average displacement from global ordering: {avg_displacement:.3f}")
    alphabet_size = len(global_order)
    print("Agreement with global ordering:")
    for k in [1, 3, 5, 10]:
        matches = 0
        total = 0
        for ordering in table.values():
            for pos in range(min(k, alphabet_size, len(ordering))):
                if ordering[pos] == global_order[pos]:
                    matches += 1
                total += 1
        agreement = 100.0 * matches / total if total else 0
        print(f"  top-{k:<2}: {agreement:.2f}%")
    distances = []
    for ordering in table.values():
        filtered = [ch for ch in ordering if ch in global_pos]
        if len(filtered) == alphabet_size:
            distances.append(kendall_distance(filtered, global_order))
    mean_distance = sum(distances) / len(distances) if distances else 0
    max_distance = alphabet_size * (alphabet_size - 1) // 2
    print(f"Average Kendall distance: {mean_distance:.2f}")
    print(f"Normalized Kendall distance: {100.0 * mean_distance / max_distance:.2f}%")
    first_letters = Counter()
    for ordering in table.values():
        if ordering:
            first_letters[ordering[0]] += 1
    print("Most common first-place letters:")
    for ch, count in first_letters.most_common(10):
        pct = 100.0 * count / len(table)
        print(f"  {repr(ch):>3} : {pct:6.2f}%")
    top5_letters = Counter()
    for ordering in table.values():
        for ch in ordering[:5]:
            top5_letters[ch] += 1
    total_slots = len(table) * 5
    print("Most common top-5 letters:")
    for ch, count in top5_letters.most_common(10):
        pct = 100.0 * count / total_slots
        print(f"  {repr(ch):>3} : {pct:6.2f}%")

def main():
    with open('input.txt', encoding='utf-8') as f:
        raw = f.read()
    stream = normalize(raw)
    alphabet = sorted(set(stream))
    global_order = build_global_freq(stream, alphabet)
    print(f"Alphabet size: {len(alphabet)}")
    print(f"Stream length: {len(stream)}")
    print(f"Global frequency order: {''.join(global_order)}")
    max_order = 4
    counts = build_ngram_counts(stream, max_order)
    static_fn = static_reorder(global_order)
    _, _, static_avg = simulate(stream, set(alphabet), static_fn, max_order)
    print(f"Static alphabet baseline: {static_avg:.4f} avg rotations/char")
    for order in range(1, max_order + 1):
        oracle_fn = make_oracle(counts, global_order, order)
        _, _, oracle_avg = simulate(stream, set(alphabet), oracle_fn, order)
        reduction = 100 * (1 - oracle_avg / static_avg)
        print(f"Oracle (max order={order}):         {oracle_avg:.4f} avg rotations/char  ({reduction:.1f}% reduction)")
        analyze_oracle(stream, oracle_fn, global_order, order)

if __name__ == '__main__':
    main()
