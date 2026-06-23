#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy

nlp = spacy.load("en_core_web_sm")


def get_child_branches_for_verb(doc, verb_input):
    """
    Đưa ra CÁC NHÁNH CON của một VERB trong dependency tree.

    - Mỗi child trực tiếp của verb là gốc của một nhánh.
    - Nhánh = toàn bộ subtree của child đó (child.subtree).
    - Không bao gồm lại verb và không đi ngược lên.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Kết quả nlp(text)
    verb_input : str hoặc spaCy Token
        - Nếu là str : tìm token VERB có text == verb_input
        - Nếu là Token: dùng trực tiếp (phải là VERB)

    Returns
    -------
    branches : List[dict]
        [
          {
            "child":      text của child,
            "child_dep":  dep của child với verb,
            "indices":    list index token trong nhánh,
            "span":       text ghép từ các token trong nhánh
          },
          ...
        ]
    """

    # 1. Xác định token VERB
    if isinstance(verb_input, str):
        verb = None
        for tok in doc:
            if tok.text == verb_input and tok.pos_ == "VERB":
                verb = tok
                break
        if verb is None:
            raise ValueError(f"Không tìm thấy VERB '{verb_input}' trong doc.")
    else:
        verb = verb_input
        if verb.pos_ != "VERB":
            raise ValueError(f"Token '{verb.text}' không phải VERB.")

    branches = []

    # 2. Mỗi child của verb là một nhánh
    for child in verb.children:
        # subtree của child = toàn bộ nhánh con
        subtree_tokens = list(child.subtree)
        subtree_tokens = sorted(subtree_tokens, key=lambda t: t.i)

        indices = [t.i for t in subtree_tokens]
        span_text = " ".join(t.text for t in subtree_tokens)

        branches.append({
            "child": child.text,
            "child_dep": child.dep_,
            "indices": indices,
            "span": span_text
        })

    return branches


if __name__ == "__main__":

    text = "The model may never have been fully tested."
    doc = nlp(text)

    print("Câu gốc:")
    print(text)
    print("-" * 80)

    # Ví dụ 2: các nhánh con của verb 'born'
    print("Các nhánh con của verb 'born':")
    branches_born = get_child_branches_for_verb(doc, "tested")
    for i, b in enumerate(branches_born, 1):
        print(f"[Branch {i}] child={b['child']:<7} dep={b['child_dep']:<8} | span: {b['span']}")
