from flask import Flask, request
from flask_cors import CORS
from scipy.sparse import load_npz
from sparse_backend import decode, encode, render_toks_w_weights, atom_query, code_query
from sparse_backend.error_handlers import (
    validate_index,
    handle_invalid_index,
    InvalidIndexError,
)
import json

app = Flask(__name__)
CORS(app)
app.register_error_handler(InvalidIndexError, handle_invalid_index)

csr_codes = load_npz("csr_codes.npz")


@app.route("/atom/<int:atom_idx>")
def get_atom(atom_idx):
    validate_index(atom_idx, csr_codes.shape[1])

    k = request.args.get("k", default=5000, type=int)
    lowest_ratio = request.args.get("lowest_ratio", default=0.14, type=float)

    code_to_weight = atom_query(atom_idx, csr_codes, k=k, lowest_ratio=lowest_ratio)

    return json.dumps(code_to_weight)


@app.route("/code/<int:code_idx>")
def get_code(code_idx):
    validate_index(code_idx, csr_codes.shape[0])

    k = request.args.get("k", default=10, type=int)
    lowest_ratio = request.args.get("lowest_ratio", default=0.1, type=float)

    atom_to_weight = code_query(code_idx, csr_codes, k=k, lowest_ratio=lowest_ratio)

    return json.dumps(atom_to_weight)


# ~~~~~~~~~~ Non-essential endpoints below ~~~~~~~~~~ #


@app.route("/see_toks/<text>")
def see_toks(text):
    return json.dumps(decode(encode(text)))


def get_atom_html(atom_idx, k=5000, lowest_ratio=0.14):
    tok_id_to_weight = atom_query(atom_idx, csr_codes, k=k, lowest_ratio=lowest_ratio)
    return render_toks_w_weights(
        list(tok_id_to_weight.keys()), list(tok_id_to_weight.values())
    )


@app.route("/render_atom/<int:atom_idx>")
def render_atom(atom_idx):
    k = request.args.get("k", default=5000, type=int)
    lowest_ratio = request.args.get("lowest_ratio", default=0.05, type=float)
    validate_index(atom_idx, csr_codes.shape[1])

    return get_atom_html(atom_idx, k=k, lowest_ratio=lowest_ratio)


def get_code_html(code_idx, k=10, lowest_ratio=0.2):
    atom_to_weight = code_query(code_idx, csr_codes, k=k, lowest_ratio=lowest_ratio)
    result = f'Code {code_idx}, "{decode([code_idx])[0]}" <br><br>'
    for atom, atom_weight in atom_to_weight.items():
        result += f"Atom {atom}, Weight {atom_weight:.3f}<br>"
        result += get_atom_html(atom, k=400, lowest_ratio=lowest_ratio)
        result += "<br>"
    return result


@app.route("/render_code/<int:code_idx>")
def render_code(code_idx):
    validate_index(code_idx, csr_codes.shape[0])

    k = request.args.get("k", default=10, type=int)
    lowest_ratio = request.args.get("lowest_ratio", default=0.2, type=float)

    return get_code_html(code_idx, k=k, lowest_ratio=lowest_ratio)


@app.route("/render_code_str/<code_str>")
def code_str_endpoint(code_str):
    code_idx = encode(code_str)[0]
    return get_code_html(code_idx)


if __name__ == "__main__":
    app.run()
