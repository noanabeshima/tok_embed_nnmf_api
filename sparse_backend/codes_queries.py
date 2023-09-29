from .error_handlers import validate_index
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

def atom_query(atom_idx, csr_codes, k=5000, lowest_ratio=0.14):
    atom_idx = int(atom_idx)
    validate_index(atom_idx, csr_codes.shape[1])
    # topk = torch.tensor(csr_codes[:, atom_idx].toarray().flatten()).topk(
    #     k=k, largest=True
    # )
    # toks = topk.indices.tolist()
    # weights = topk.values.tolist()
    toks, weights = topk(csr_codes[:, atom_idx].toarray().flatten(), k=k)

    res = {
        tok: weight
        for tok, weight in zip(toks, weights)
        if weight / weights[0] > lowest_ratio
    }
    return res

def atom_query(atom_idx, csr_codes, k=5000, lowest_ratio=0.14):
    atom_idx = int(atom_idx)
    validate_index(atom_idx, csr_codes.shape[1])

    # topk = torch.tensor(csr_codes[:, atom_idx].toarray().flatten()).topk(
    #     k=k, largest=True
    # )
    # toks = topk.indices.tolist()
    # weights = topk.values.tolist()
    toks, weights = topk(csr_codes[:, atom_idx].toarray().flatten(), k=k)

    res = {
        tok: weight
        for tok, weight in zip(toks, weights)
        if weight / weights[0] > lowest_ratio
    }
    return res


def code_query(code_idx: int, csr_codes, k=10, lowest_ratio=0.1):
    validate_index(code_idx, csr_codes.shape[0])

    # topk = torch.tensor(csr_codes[code_idx].toarray().flatten()).topk(k=k, largest=True)
    # atoms = topk.indices.tolist()
    # weights = topk.values.tolist()
    atoms, weights = topk(csr_codes[code_idx].toarray().flatten(), k=k)

    return {
        atom: weight
        for atom, weight in zip(atoms, weights)
        if weight / weights[0] > lowest_ratio
    }
