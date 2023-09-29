from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("EleutherAI/pythia-70m-v0")


def decode(tok_id_list, postprocessing=False):
    tok_strs = [tokenizer.decode([tok]) for tok in tok_id_list]
    if postprocessing:
        tok_strs = [tok.replace(" ", "○").replace("\n", "↵") for tok in tok_strs]
    return tok_strs


def encode(text):
    return tokenizer.encode(text).ids


def render_toks_w_weights(toks, weights):
    toks = decode(toks, postprocessing=True)

    highlighted_text = []

    for weight, tok in zip(weights, toks):
        if weight > 0.0:
            highlighted_text.append(
                f'<span style="background-color:rgba(135,206,250,{min(1.3*weight, 1)});border: 0.3px solid black;padding: 0.3px">{tok}</span>'
            )
    highlighted_text = " ".join(highlighted_text)
    highlighted_text = f'<div style="width: 75%">{highlighted_text}</div>'

    return highlighted_text
