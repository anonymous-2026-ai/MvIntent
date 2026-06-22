import math

def run_pmi_pipeline_v2(file_path):
    # 1. Định nghĩa danh mục các nhãn intent cố định
    intents = ["motivation", "background", "uses", "extends", "similarities", "differences", "future_work"]
    n_labels = len(intents)
    
    # Khởi tạo bộ đếm dữ liệu bằng mảng (List of Lists)
    single_counts = [0] * n_labels
    co_counts = [[0] * n_labels for _ in range(n_labels)]
    total_contexts = 0  # Tổng số ngữ cảnh (dòng) hợp lệ được xử lý

    # 2. Duyệt file xử lý dữ liệu
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Tách chuỗi theo ký tự đặc biệt <fff>
            parts = [p.strip() for p in line.split('<fff>') if p.strip()]
            num_parts = len(parts)
            
            # Chỉ xử lý các dòng có 1 cụm hoặc 2 cụm CV
            if num_parts not in [1, 2]:
                continue
                
            # Trích xuất vị trí các intent vượt ngưỡng > 0.5 cho từng cụm
            intents_part1 = [i for i, v in enumerate(parts[0].split()) if float(v) > 0.5]
            intents_part2 = [i for i, v in enumerate(parts[1].split()) if float(v) > 0.5] if num_parts == 2 else []
            
            # Kiểm tra tính hợp lệ của dòng để đưa vào thống kê
            is_valid = False
            if num_parts == 1 and intents_part1:
                is_valid = True
            elif num_parts == 2 and intents_part1 and intents_part2:
                is_valid = True
                
            if is_valid:
                total_contexts += 1
                
                # --- THỐNG KÊ TẦN SUẤT ĐƠN LẺ ---
                # Gom toàn bộ các intent xuất hiện trên dòng này lại (loại bỏ trùng lặp nếu cần)
                all_intents_on_line = list(set(intents_part1 + intents_part2))
                for idx in all_intents_on_line:
                    single_counts[idx] += 1
                
                # --- THỐNG KÊ ĐỒNG XUẤT HIỆN ---
                # Trường hợp 1: Đồng xuất hiện giữa các cụm (chỉ có ở dòng có 2 cụm)
                if num_parts == 2:
                    for idx1 in intents_part1:
                        for idx2 in intents_part2:
                            co_counts[idx1][idx2] += 1
                            if idx1 != idx2:
                                co_counts[idx2][idx1] += 1
                
                # Trường hợp 2: Đồng xuất hiện nội tại trong cùng 1 cụm (nếu cụm đó có đa nhãn > 0.5)
                # Áp dụng cho cả cụm 1 (của dòng 1 CV và 2 CV) và cụm 2 (của dòng 2 CV)
                for current_part_intents in [intents_part1, intents_part2]:
                    if len(current_part_intents) >= 2:
                        for i in range(len(current_part_intents)):
                            for j in range(i + 1, len(current_part_intents)):
                                idx1 = current_part_intents[i]
                                idx2 = current_part_intents[j]
                                co_counts[idx1][idx2] += 1
                                co_counts[idx2][idx1] += 1

    # 3. Tính toán giá trị ma trận PMI từ các bộ đếm
    pmi_matrix = [[None] * n_labels for _ in range(n_labels)]
    
    if total_contexts == 0:
        print("Không có dòng nào thỏa mãn điều kiện dữ liệu.")
        return intents, pmi_matrix

    for i in range(n_labels):
        # Xác suất biên P(A) dựa trên tổng số ngữ cảnh hợp lệ
        p_i = single_counts[i] / total_contexts
        
        for j in range(n_labels):
            p_j = single_counts[j] / total_contexts
            # Xác suất đồng xuất hiện thực tế P(A, B)
            p_ij = co_counts[i][j] / total_contexts
            
            if p_ij > 0 and p_i > 0 and p_j > 0:
                pmi_matrix[i][j] = round(math.log2(p_ij / (p_i * p_j)), 4)
            else:
                pmi_matrix[i][j] = "-inf"
                
    return intents, pmi_matrix

def print_pmi_table(intents, pmi_matrix):
    header = f"{'Intent':<15}" + "".join([f"{intent:>15}" for intent in intents])
    print(header)
    print("-" * len(header))
    
    for i, intent_i in enumerate(intents):
        row_str = f"{intent_i:<15}"
        for j in range(len(intents)):
            val = pmi_matrix[i][j]
            if isinstance(val, float):
                row_str += f"{val:>15.4f}"
            else:
                row_str += f"{str(val)::>15}"
        print(row_str)

# --- Cách thực thi ---
# intents, pmi_res = run_pmi_pipeline_v2("1780900164.9749525/triple_intent_att_5200.txt")
# print_pmi_table(intents, pmi_res)

def read_file_result(filein , fileout):
    intents = ["motivation", "background", "uses", "extends", "similarities", "differences", "future_work"]

    with open(filein, 'r') as f, open(fileout , 'w') as f_w:
        all_result = [] 
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Tách chuỗi theo ký tự đặc biệt <fff>
            list_part = [p.strip() for p in line.split('<fff>') if p.strip()]

            list_intent = [] 
            for part in list_part:
                # print(part)
                list_num  = [float(t) for t in part.split()]
                for i, num in enumerate(list_num):
                    if num > 0.5:
                        list_intent.append(intents[i])
            if len(list_intent) > 0 : 
                all_result.append(list(set(list_intent)))
        f_w.write("\n".join( [ " ".join(all_result[i])  for i in range(len(all_result))] ))

# read_file_result('1780900164.9749525/triple_intent_att_5200.txt','test_pmi.txt')

def compute_pmi_pair(intent1, intent2, list_intent):
    count1 , count2 , count12 = 0 , 0 , 0 
    for intents in list_intent:
        if intent1 in intents:
            count1+=1 
        if intent2 in intents:
            count2 += 1
        if intent1 in intents and intent2 in intents:
            count12 += 1 
    return math.log2(count12 * len(list_intent) / count1 / count2)



if __name__ == '__main__':
    list_intent = [] 
    with open('test_pmi.txt', 'r') as f:
        for line in f:
            line = line.strip()
            list_intent.append([t.strip() for t in line.split()])
    pmi_pair = compute_pmi_pair('uses','background', list_intent)
    print(pmi_pair)