#!/usr/bin/env python
"""
Vẽ dependency tree theo cấu trúc cây, KHÔNG bao gồm punctuation:
- Node  = token (từ) nhưng bỏ toàn bộ dấu câu
- Edge  = quan hệ phụ thuộc
- Layout dạng cây: root ở trên, con ở dưới
- ĐÃ GIÃN CÁCH NODE để đỡ đè lên nhau
"""

import spacy
import matplotlib.pyplot as plt


def compute_depth(token):
    """Tính depth của token trong dependency tree."""
    if token.head == token:  # ROOT
        return 0
    return 1 + compute_depth(token.head)


def draw_dep_tree(text: str):
    """Vẽ dependency tree + bỏ punctuation, giãn node, và in toàn bộ dependency."""
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    sent = list(doc.sents)[0]

    print("\n===== ALL DEPENDENCIES (NO PUNCTUATION) =====")
    for token in sent:
        # if token.pos_ == "PUNCT":
        #     continue
        
        head = token.head
        rel = token.dep_
        if rel == "ROOT":
            print(f"{token.text}  --ROOT-->  (ROOT)")
        else:
            # Nếu head là punctuation → bỏ
            if head.pos_ != "PUNCT":
                print(f"{token.text}  --{rel}-->  {head.text}")

    print("=============================================\n")

    # 1. Bỏ punctuation
    tokens = [token for token in sent]

    # Map lại chỉ số từ cũ → chỉ số mới sau khi loại bỏ punct
    index_map = {tok.i: new_i for new_i, tok in enumerate(tokens)}

    # 2. Tính depth
    depths = {index_map[token.i]: compute_depth(token) for token in tokens}

    # Hệ số giãn cách
    x_step = 3.0
    y_step = 2.0

    # 3. Tọa độ
    positions = {
        index_map[token.i]: (
            index_map[token.i] * x_step,
            -depths[index_map[token.i]] * y_step
        )
        for token in tokens
    }

    fig_width = max(12, len(tokens) * 0.8)
    plt.figure(figsize=(fig_width, 6))

    # 4. Vẽ các cạnh
    for token in tokens:
        if token.dep_ == "ROOT":
            continue

        if token.head.pos_ == "PUNCT":
            continue

        child_id = index_map[token.i]
        head_id = index_map[token.head.i]

        x_child, y_child = positions[child_id]
        x_head, y_head = positions[head_id]

        plt.plot([x_head, x_child], [y_head, y_child], linewidth=1)

        x_mid = (x_head + x_child) / 2
        y_mid = (y_head + y_child) / 2
        plt.text(
            x_mid,
            y_mid + 0.15,
            token.dep_,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    # 5. Vẽ node
    for token in tokens:
        node_id = index_map[token.i]
        x, y = positions[node_id]

        plt.scatter([x], [y], s=500)

        label = f"{token.text}\n{token.pos_}"
        plt.text(
            x,
            y,
            label,
            fontsize=9,
            ha="center",
            va="center",
        )

    plt.title("Dependency Tree (Without Punctuation)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open('ner_example.txt', 'r') as f:
        text = f.read()
    draw_dep_tree(text)