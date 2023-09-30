from .error_handlers import validate_index
from .rendering import tokenizer
import numpy as np
from scipy.sparse import load_npz
from functools import lru_cache


def topk(arr, k=10):
    assert isinstance(k, int) and k > 0

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    assert len(arr.shape) == 1

    indices = arr.argsort()[-k:][::-1]
    vals = arr[indices]
    return indices.tolist(), vals.tolist()

csr_codes = load_npz("csr_codes.npz")
csc_codes = load_npz("csc_codes.npz")


@lru_cache(maxsize=5000)
def atom_query(atom_idx, k=5000, lowest_ratio=0.14, string=False):
    atom_idx = int(atom_idx)
    validate_index(atom_idx, csr_codes.shape[1])

    indices = csc_codes[:, atom_idx].indices
    _idx, weights = topk(csc_codes[:, atom_idx].data, k=k)
    tok_ids = indices[_idx]
    

    if string:
        res = [
            (tokenizer.decode([tok]), weight)
            for tok, weight in zip(tok_ids, weights)
            if weight / (weights[0]+1e-3) > lowest_ratio
        ]
    else:
        res = {
            tok: weight
            for tok, weight in zip(tok_ids, weights)
            if weight / (weights[0]+1e-3) > lowest_ratio
        }
    return res


def code_query(code_idx: int, csr_codes, k=10, lowest_ratio=0.1):
    validate_index(code_idx, csr_codes.shape[0])

    atoms, weights = topk(csr_codes[code_idx].toarray().flatten(), k=k)

    return [
        (atom, weight)
        for atom, weight in zip(atoms, weights)
        if weight / (weights[0]+1e-3) > lowest_ratio
    ]
