import spacy
nlp = spacy.load("en_core_web_sm")


def build_verb_phrase_general(verb):
    """
    Xây cụm động từ (verb phrase) từ 1 VERB 'verb' một cách tổng quát:
    - verb chính
    - + các child có dep_ thuộc: aux, auxpass, neg, prt
    - + các child có pos_ là AUX hoặc PART
    - KHÔNG dùng span min-max, mà join trực tiếp các token trong group.
    """
    if verb.pos_ != "VERB":
        raise ValueError(f"Token '{verb.text}' không phải VERB.")

    sent = verb.sent

    VERB_DEP_ROLES = {"aux", "auxpass", "neg", "prt"}

    group = {verb}

    changed = True
    while changed:
        changed = False
        for tok in sent:
            # Nếu head của tok thuộc group → xét xem có nên thêm
            if tok.head in group and tok not in group:
                if tok.dep_ in VERB_DEP_ROLES:
                    group.add(tok)
                    changed = True
                    continue

                # AUX / PART trực tiếp gắn vào verb-phrase
                if tok.pos_ in {"AUX", "PART"}:
                    group.add(tok)
                    changed = True
                    continue

    # Sắp theo thứ tự và join đúng các token trong group
    group_tokens = sorted(group, key=lambda t: t.i)
    phrase = " ".join(tok.text for tok in group_tokens)

    return phrase, group_tokens



def get_child_branches_for_verb(verb):
    """
    Lấy TẤT CẢ các nhánh con của một VERB:
    - Mỗi child trực tiếp của verb là gốc của một nhánh.
    - Nhánh = toàn bộ subtree của child đó (child.subtree).
    - Loại bỏ các child đã được gộp vào verb phrase (aux, auxpass, neg, prt, AUX, PART).

    Parameters
    ----------
    verb : spaCy Token (pos_ == "VERB")

    Returns
    -------
    branches : List[dict]
        [
          {
            "child":      text của child (gốc nhánh),
            "child_dep":  dep của child với verb,
            "indices":    list index token trong nhánh,
            "span":       text ghép từ các token trong nhánh
          },
          ...
        ]
    """
    if verb.pos_ != "VERB":
        raise ValueError(f"Token '{verb.text}' không phải VERB.")

    # Lấy verb phrase để biết những token nào đã gộp vào cụm động từ
    _, vp_tokens = build_verb_phrase_general(verb)
    vp_set = set(vp_tokens)

    branches = []

    for child in verb.children:
        # Nếu child đã nằm trong verb phrase (aux/neg/...) thì bỏ qua, không coi là một nhánh riêng
        if child in vp_set:
            continue

        subtree_tokens = list(child.subtree)
        subtree_tokens = sorted(subtree_tokens, key=lambda t: t.i)

        indices = [t.i for t in subtree_tokens]
        span_text = " ".join(t.text for t in subtree_tokens)

        branches.append({
            "child": child.text,
            "child_dep": child.dep_,
            "indices": indices,
            "span": span_text,
        })

    return branches
    
def summarize_verb_with_subject_branches(verb):
    """
    Nhận đầu vào là VERB + thông tin các nhánh của nó,
    trả về:
      - verb_phrase tổng quát (dùng build_verb_phrase_general)
      - tập các nhánh chứa chủ ngữ của động từ đó

    Chủ ngữ được xác định bằng dep_:
      nsubj, nsubjpass, csubj, csubjpass
    (nếu cần, bạn có thể mở rộng set này sau)
    """
    # 1. Cụm động từ tổng quát
    verb_phrase, _ = build_verb_phrase_general(verb)

    # 2. Tất cả nhánh con (trừ phần verb phrase)
    branches = get_child_branches_for_verb(verb)

    # 3. Lọc các nhánh là chủ ngữ
    SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
    subject_branches = [b for b in branches if b["child_dep"] in SUBJ_DEPS]

    result = {
        "verb_head": verb.text,
        "verb_phrase": verb_phrase,
        "branches": branches,
        "subject_branches": subject_branches,
    }
    return result

if __name__ == "__main__":
    text = (
        "Recently impressive results have been reported for representation learning, that generalizes to different downstream tasks, through self-supervised learning for text and speech (Devlin et al., 2018; Baevski et al., 2019a; van den Oord et al., 2018; Baevski et al., 2019b) ."
    )
    doc = nlp(text)

    print("Sentence:", text)
    print("-" * 100)

    for tok in doc:
        if tok.pos_ == "VERB":
            info = summarize_verb_with_subject_branches(tok)
            print(f"\n=== VERB: {info['verb_head']} ===")
            print("Verb phrase:", info["verb_phrase"])

            print("All branches:")
            for b in info["branches"]:
                print(f"  - child={b['child']:<10} dep={b['child_dep']:<10} | span: {b['span']}")

            print("Subject branches:")
            for b in info["subject_branches"]:
                print(f"  * child={b['child']:<10} dep={b['child_dep']:<10} | span: {b['span']}")