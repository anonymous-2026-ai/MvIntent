import spacy
from collections import deque

def build_directed_adj(doc):
    """
    Xây đồ thị CÓ HƯỚNG từ dependency tree:
    - Node: token.i
    - Cạnh: head --> child
    - KHÔNG bỏ các cạnh 'obj' nữa
    """
    adj = {token.i: set() for token in doc}

    for token in doc:
        # bỏ ROOT (head chính là nó)
        if token.head == token:
            continue

        i = token.head.i   # từ head
        j = token.i        # tới child

        # đồ thị có hướng: head -> child
        adj[i].add(j)

    return adj

def find_nearest_noun(doc, adj, start_i):
    """
    Tìm node danh từ (NOUN/PRON) gần nhất tính theo số cạnh
    trong đồ thị (giả sử vô hướng), nhưng:
    - BỎ QUA token chứa "CITATION"
    - CHỈ CHẤP NHẬN những node mà đường đi tới nó
      có ÍT NHẤT 1 cạnh có quan hệ chứa 'obj'
    """
    start = doc[start_i]

    # (node_i, dist, has_obj_edge_on_path)
    q = deque([(start_i, 0, False)])
    visited = set([(start_i, False)])  # xét cả trạng thái has_obj

    while q:
        node_i, dist, has_obj = q.popleft()
        token = doc[node_i]

        # BỎ QUA token chứa "CITATION" trong text (chỉ không lấy làm kết quả)
        if "CITATION" not in token.text.upper():
            # Nếu là noun/pronoun và đã có ít nhất 1 cạnh 'obj' trên đường đi
            if node_i != start_i and token.pos_ in {"NOUN", "PROPN"} and has_obj:
                return token, dist

        # BFS mở rộng hàng xóm
        for nb in adj[node_i]:
            # Xác định xem cạnh node_i <-> nb có phải là một quan hệ chứa 'obj' không
            edge_has_obj = False

            # child là đứa có head trỏ sang node còn lại
            if doc[nb].head.i == node_i:
                # quan hệ là dep_ của child
                if "obj" in doc[nb].dep_:
                    edge_has_obj = True
            elif doc[node_i].head.i == nb:
                if "obj" in doc[node_i].dep_:
                    edge_has_obj = True

            new_has_obj = has_obj or edge_has_obj

            state = (nb, new_has_obj)
            if state not in visited:
                visited.add(state)
                q.append((nb, dist + 1, new_has_obj))

    return None, None

def extract_object(nlp, text):
    doc = nlp(text)
    adj = build_directed_adj(doc)

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
# objects  = extract_object(nlp , text)
# for verb in objects.keys():
#     print(verb , verb.i , objects[verb].i, objects[verb].text)