

with open('1781508488.021478/triple_weight_200.txt' ,'r') as f_r:
    num_intent = [] 
    for line in f_r:
        line = line.strip()
        num_intent.append(len(line.split()))

# print(num_intent)

def calculate_distribution(number_list):
    # 1. Đếm tần suất xuất hiện bằng Dictionary
    frequency_dict = {}
    for num in number_list:
        frequency_dict[num] = frequency_dict.get(num, 0) + 1
        
    total_count = len(number_list)
    
    # 2. Sắp xếp lại theo thứ tự tăng dần của các số
    sorted_numbers = sorted(frequency_dict.keys())
    
    # 3. Tính toán phân phối phần trăm và lưu dạng List của List
    # Mỗi phần tử con: [Số, Số lần xuất hiện, Tỷ lệ phần trăm]
    distribution_result = []
    for num in sorted_numbers:
        count = frequency_dict[num]
        percentage = (count / total_count) * 100
        distribution_result.append([num, count, round(percentage, 2)])
        
    return distribution_result

# --- Ví dụ minh họa với một phần dữ liệu mô phỏng từ ảnh của bạn ---
raw_data = [1, 2, 1, 2, 7, 2, 7, 2, 2, 1, 3, 1, 1, 4, 2, 2, 3, 5, 1, 1, 8, 7, 3, 4, 8]

dist = calculate_distribution(num_intent)

# In kết quả dạng bảng scannable
print(f"{'Giá trị (Value)':<15} | {'Tần suất (Count)':<15} | {'Tỷ lệ (Percentage)':<15}")
print("-" * 55)
for row in dist:
    print(row[0] , row[2])