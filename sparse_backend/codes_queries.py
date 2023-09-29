import torch
import json


def atom_query(atom_idx, csr_codes, k=5000, lowest_ratio=0.14):
    atom_idx = int(atom_idx)
    if atom_idx < 0 or atom_idx >= csr_codes.shape[1]:
        return (
            "Invalid atom index. Must be between 0 and "
            + str(csr_codes.shape[1] - 1)
            + " inclusive.",
            status.HTTP_400_BAD_REQUEST,
        )

    topk = torch.tensor(csr_codes[:, atom_idx].toarray().flatten()).topk(
        k=k, largest=True
    )
    toks = topk.indices.tolist()
    weights = topk.values.tolist()

    res = {
        tok: weight
        for tok, weight in zip(toks, weights)
        if weight / weights[0] > lowest_ratio
    }
    return res


def code_query(code_idx: int, csr_codes, k=10, lowest_ratio=0.1):
    topk = torch.tensor(csr_codes[code_idx].toarray().flatten()).topk(k=k, largest=True)
    atoms = topk.indices.tolist()
    weights = topk.values.tolist()

    return {
        atom: weight
        for atom, weight in zip(atoms, weights)
        if weight / topk.values[0] > lowest_ratio
    }
