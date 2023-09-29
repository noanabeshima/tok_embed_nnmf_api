from .error_handlers import validate_index
from .rendering import tokenizer
import numpy as np

def topk(arr, k=10):
    assert len(arr.shape) == 1
    assert isinstance(k, int) and k > 0

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    assert len(arr.shape) == 1

    indices = arr.argsort()[-k:][::-1]
    vals = arr[indices]
    return indices.tolist(), vals.tolist()


def atom_query(atom_idx, csr_codes, k=5000, lowest_ratio=0.14, string=False):
    atom_idx = int(atom_idx)
    validate_index(atom_idx, csr_codes.shape[1])

    toks, weights = topk(csr_codes[:, atom_idx].toarray().flatten(), k=k)

    res = {
        tok: weight
        for tok, weight in zip(toks, weights)
        if weight / (weights[0]+1e-3) > lowest_ratio
    }

    if string:
        res = [
            {'tok': tokenizer.decode([tok]), 'weight': weight, 'tok_id': tok}
            for tok, weight in res.items()
        ]
    return res


def code_query(code_idx: int, csr_codes, k=10, lowest_ratio=0.1):
    validate_index(code_idx, csr_codes.shape[0])

    atoms, weights = topk(csr_codes[code_idx].toarray().flatten(), k=k)

    return [
        {'atom': atom, 'weight': weight}
        for atom, weight in zip(atoms, weights)
        if weight / (weights[0]+1e-3) > lowest_ratio
    ]
