import spacy
from collections import deque


# Xây adjacency list cho ĐỒ THỊ VÔ HƯỚNG từ dependency tree
def build_undirected_adj(doc):
    adj = {token.i: set() for token in doc}

    for token in doc:
        # bỏ ROOT
        if token.head == token:
            continue

        # BỎ QUA mọi quan hệ chứa "obj": obj, dobj, iobj, pobj, cobj,...
        if "obj" in token.dep_:
            continue

        i = token.i
        j = token.head.i

        # đồ thị vô hướng
        adj[i].add(j)
        adj[j].add(i)

    return adj

def find_nearest_noun(doc, adj, start_i):
    """
    Tìm node danh từ (NOUN/PROPN) gần nhất tính theo số cạnh
    trong đồ thị VÔ HƯỚNG, nhưng bỏ qua token chứa "CITATION".
    """
    start = doc[start_i]
    q = deque([(start_i, 0)])
    visited = {start_i}

    while q:
        node_i, dist = q.popleft()
        token = doc[node_i]

        # BỎ QUA token chứa "CITATION" trong text
        if "CITATION" in token.text.upper():     # hoặc token.text để phân biệt hoa/thường
            # không return, chỉ skip node này
            pass
        else:
            # Nếu là noun/proper noun → trả về
            if node_i != start_i and token.pos_ in {"NOUN", "PRON"}:
                return token, dist

        # BFS mở rộng hàng xóm
        for nb in adj[node_i]:
            if nb not in visited:
                visited.add(nb)
                q.append((nb, dist + 1))

    return None, None

def extract_subject(nlp, text):
    doc = nlp(text)
    adj = build_undirected_adj(doc)

    list_verb_token = [token for token in doc if token.pos_ == "VERB"]
    out = {}
    for verb_token  in list_verb_token:
        nearest_noun, distance = find_nearest_noun(doc, adj, verb_token.i)
        if nearest_noun != None : 
            out[verb_token.i] = nearest_noun.i 
        else:
            out[verb_token.i]= None 
    return out 

# nlp = spacy.load("en_core_web_sm")
# # with open('ner_example.txt' ,'r') as f_r:
# #     text = f_r.read()
# # list_subject= extract_subject(text)
# # print(list_subject)
# text = "we study math."
# # doc = nlp(text)
# # adj = build_directed_adj(doc)
# # print(adj)
# objects  = extract_subject(nlp , text)
# for verb in objects.keys():
#     print(verb , verb.i , objects[verb].i, objects[verb].text)