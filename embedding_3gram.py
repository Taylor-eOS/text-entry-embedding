import math
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OMIT = frozenset("qx")
SENTINEL = "^"
text_path = "input.txt"
context_len = 3
dims = [2, 4, 8, 16, 32]
epochs = 40
batch_size = 512
lr = 0.01
val_fraction = 0.1
seed = 1
random.seed(seed)
torch.manual_seed(seed)

def normalize(raw, omit=OMIT):
    out = []
    for ch in raw:
        if ch in (" ", "\n"):
            out.append(" ")
        elif ch.isalpha():
            low = ch.lower()
            if low not in omit:
                out.append(low)
    return out

def build_vocab(stream):
    alphabet = sorted(set(stream))
    if SENTINEL in alphabet:
        alphabet.remove(SENTINEL)
    alphabet = [SENTINEL] + alphabet
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return alphabet, char_to_idx, idx_to_char

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
        if ch in alphabet and ch != SENTINEL:
            freq[ch] += 1
    total = sum(freq.values()) or 1
    return sorted([ch for ch in alphabet if ch != SENTINEL], key=lambda c: -freq[c] / total)

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

def static_reorder(global_order):
    def reorder(context):
        return global_order
    return reorder

def simulate(stream, alphabet, reorder_fn, context_len):
    total_cost = 0
    total_chars = 0
    padding = [SENTINEL] * context_len
    padded = padding + stream
    candidate_set = [ch for ch in alphabet if ch != SENTINEL]
    for i in range(context_len, len(padded)):
        ch = padded[i]
        if ch not in candidate_set:
            continue
        context = padded[i - context_len:i]
        ordering = reorder_fn(context)
        try:
            pos = ordering.index(ch)
        except ValueError:
            pos = len(ordering)
        total_cost += pos
        total_chars += 1
    avg = total_cost / total_chars if total_chars else 0.0
    return total_cost, total_chars, avg

def make_examples(stream, context_len, char_to_idx):
    padded = [SENTINEL] * context_len + stream
    x = []
    y = []
    for i in range(context_len, len(padded)):
        tgt = padded[i]
        if tgt not in char_to_idx:
            continue
        ctx = padded[i - context_len:i]
        if any(ch not in char_to_idx for ch in ctx):
            continue
        x.append([char_to_idx[ch] for ch in ctx])
        y.append(char_to_idx[tgt])
    if not x:
        raise ValueError("No training examples were created from the input text.")
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def expected_rank_loss(logits, targets, sentinel_idx):
    batch = logits.size(0)
    vocab = logits.size(1)
    mask = torch.ones(vocab, dtype=torch.bool, device=logits.device)
    mask[sentinel_idx] = False
    valid_logits = logits[:, mask]
    valid_size = valid_logits.size(1)
    full_to_valid = torch.full((vocab,), -1, dtype=torch.long, device=logits.device)
    valid_indices = torch.where(mask)[0]
    full_to_valid[valid_indices] = torch.arange(valid_size, device=logits.device)
    target_valid = full_to_valid[targets]
    target_scores = valid_logits[torch.arange(batch, device=logits.device), target_valid].unsqueeze(1)
    margins = valid_logits - target_scores
    loss = torch.sigmoid(margins).sum(dim=1) - torch.sigmoid(torch.zeros(1, device=logits.device))
    return loss.mean()

class CharOrderModel(nn.Module):
    def __init__(self, vocab_size, dim, context_len, sentinel_idx):
        super().__init__()
        self.context_len = context_len
        self.sentinel_idx = sentinel_idx
        self.char_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(context_len, dim)
        self.mix = nn.Linear(dim, dim, bias=False)
        self.output_emb = nn.Embedding(vocab_size, dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.eye_(self.mix.weight)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pos_ids = torch.arange(self.context_len, device=x.device)
        char_vecs = self.char_emb(x)
        pos_vecs = self.pos_emb(pos_ids).unsqueeze(0)
        state = (char_vecs + pos_vecs).sum(dim=1)
        state = self.mix(state)
        logits = state @ self.output_emb.weight.t() + self.output_bias
        logits = logits.clone()
        logits[:, self.sentinel_idx] = -1e9
        return logits

    def predict_ordering(self, context_ids, idx_to_char):
        device = next(self.parameters()).device
        x = torch.tensor(context_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.forward(x).squeeze(0)
            order = torch.argsort(logits, descending=True).tolist()
        return [idx_to_char[i] for i in order if i != self.sentinel_idx]

    def parameter_bytes(self):
        return sum(p.numel() for p in self.parameters()) * 4

    def runtime_bytes(self):
        state_bytes = (self.char_emb.embedding_dim) * 4
        return state_bytes

def train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = expected_rank_loss(logits, yb, model.sentinel_idx)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_seen += xb.size(0)
        train_loss = total_loss / max(total_seen, 1)
        val_loss = evaluate_rank_loss(model, val_x, val_y, batch_size)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_rank_loss(model, x, y, batch_size):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = expected_rank_loss(logits, yb, model.sentinel_idx)
            total_loss += loss.item() * xb.size(0)
            total_seen += xb.size(0)
    return total_loss / max(total_seen, 1)

def evaluate_rotations(model, x, y, idx_to_char, char_to_idx, batch_size):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    valid_indices = [i for i in range(len(idx_to_char)) if i != char_to_idx[SENTINEL]]
    valid_indices_t = torch.tensor(valid_indices, dtype=torch.long)
    full_to_valid = torch.full((len(idx_to_char),), -1, dtype=torch.long)
    for pos, idx in enumerate(valid_indices):
        full_to_valid[idx] = pos
    model.eval()
    total_cost = 0
    total_seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            valid_logits = logits.index_select(1, valid_indices_t.to(logits.device))
            order = torch.argsort(valid_logits, dim=1, descending=True)
            positions = torch.empty_like(order)
            ranks = torch.arange(order.size(1), device=order.device).unsqueeze(0).expand_as(order)
            positions.scatter_(1, order, ranks)
            target_cols = full_to_valid[yb].to(order.device)
            target_pos = positions.gather(1, target_cols.unsqueeze(1)).squeeze(1)
            total_cost += int(target_pos.sum().item())
            total_seen += xb.size(0)
    avg = total_cost / total_seen if total_seen else 0.0
    return total_cost, total_seen, avg

def run_dim_sweep(dims, alphabet, char_to_idx, idx_to_char, train_x, train_y, val_x, val_y, val_stream, context_len, epochs, batch_size, lr, seed, static_avg, oracle_avg,):
    print(f"\n{'dim':>5}  {'rotations/char':>14}  {'vs static':>10}  {'vs oracle':>10}  {'flash B':>10}  {'runtime B':>10}")
    print("-" * 67)
    results = []
    for dim in dims:
        model = CharOrderModel(len(alphabet), dim, context_len, char_to_idx[SENTINEL])
        print(f"\ndim={dim}")
        model = train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed)
        cost, chars, avg = evaluate_rotations(model, val_x, val_y, idx_to_char, char_to_idx, batch_size)
        flash = model.parameter_bytes()
        runtime = model.runtime_bytes()
        vs_static = static_avg - avg
        vs_oracle = oracle_avg - avg
        results.append((dim, avg, flash, runtime))
        print(f"\n{dim:>5}  {avg:>14.4f}  {vs_static:>+10.4f}  {vs_oracle:>+10.4f}  {flash:>10}  {runtime:>10}")
    return results

def main():
    with open(text_path, encoding="utf-8") as f:
        raw = f.read()
    stream = normalize(raw)
    alphabet, char_to_idx, idx_to_char = build_vocab(stream)
    if len(alphabet) <= 2:
        raise ValueError("The input text does not contain enough characters after normalization.")
    split = max(int(len(stream) * (1 - val_fraction)), context_len + 1)
    split = min(split, len(stream) - 1)
    train_stream = stream[:split]
    val_stream = stream[split - context_len:]
    train_x, train_y = make_examples(train_stream, context_len, char_to_idx)
    val_x, val_y = make_examples(val_stream, context_len, char_to_idx)
    global_order = build_global_freq(train_stream, alphabet)
    counts = build_ngram_counts([SENTINEL] * context_len + train_stream, context_len)
    static_fn = static_reorder(global_order)
    oracle_fn = make_oracle(counts, global_order, context_len)
    static_cost, static_chars, static_avg = simulate(val_stream, alphabet, static_fn, context_len)
    oracle_cost, oracle_chars, oracle_avg = simulate(val_stream, alphabet, oracle_fn, context_len)
    print(f"Alphabet size:      {len(alphabet) - 1}")
    print(f"Context length:     {context_len}")
    print(f"Train examples:     {len(train_x)}")
    print(f"Validation examples:{len(val_x)}")
    print(f"Static baseline:    {static_avg:.4f} rotations/char")
    print(f"Oracle {context_len}-gram:     {oracle_avg:.4f} rotations/char")
    results = run_dim_sweep( dims, alphabet, char_to_idx, idx_to_char, train_x, train_y, val_x, val_y, val_stream, context_len, epochs, batch_size, lr, seed, static_avg, oracle_avg,)
    print("\n\nDimension sweep summary:")
    print(f"{'dim':>5}  {'rotations/char':>14}  {'flash B':>10}  {'runtime B':>10}")
    print("-" * 45)
    for dim, avg, flash, runtime in results:
        print(f"{dim:>5}  {avg:>14.4f}  {flash:>10}  {runtime:>10}")
    print(f"\nStatic baseline: {static_avg:.4f}")
    print(f"Oracle {context_len}-gram:  {oracle_avg:.4f}")
    best_dim = min(results, key=lambda r: r[1])[0]
    print(f"\nBest dim by rotations/char: {best_dim}")
    model = CharOrderModel(len(alphabet), best_dim, context_len, char_to_idx[SENTINEL])
    model = train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed)
    sample = val_stream[:context_len + 12]
    if len(sample) >= context_len:
        context = sample[:context_len]
        ordering = model.predict_ordering([char_to_idx[c] for c in context], idx_to_char)
        print(f"\nSample context (best dim={best_dim}): {''.join(context)}")
        print(f"Top 10 predictions: {''.join(ordering[:10])}")

if __name__ == "__main__":
    main()
