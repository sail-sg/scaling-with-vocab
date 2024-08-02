def find_multiple(n: int, k=128, tp=1) -> int:
    '''
    The function to resize the vocabulary size (n) to a multiple of (k).
    Typically we set k as 128 for compatibility with NVIDIA's tensor core to accelerate matrix
    multiplication. 
    The 'tp' is the size of tensor parallel in case we use  tensor parallel in distributed training.
    '''    
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