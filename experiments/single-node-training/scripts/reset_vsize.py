def find_multiple(n: int, k=128, tp=8) -> int:
    assert k > 0
    if tp ==1:
        if n % k == 0:
            return n
        return n + k - (n % k)
    elif tp > 1:
        n = n // tp
        if n % k == 0:
            return n * tp
        return (n + k - (n % k) ) * tp

for vocab_size in [1000]:
    print(n=vocab_size, find_multiple(n,tp=8))