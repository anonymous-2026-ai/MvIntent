from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# 1. Dữ liệu dạng index của bạn
# Giả sử có tổng cộng 4 nhãn (0, 1, 2, 3)
y_true_indices = [[0, 2], [1], [0, 2, 3]]
y_pred_indices = [[0], [1, 2], [0, 2]]

# 2. Khởi tạo và khớp MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3]) 

# Chuyển đổi sang ma trận nhị phân (0-1 matrix)
y_true_bin = mlb.fit_transform(y_true_indices)
y_pred_bin = mlb.fit_transform(y_pred_indices)

# Bây giờ bạn có thể tính toán như bình thường
macro_p = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
macro_r = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)

print(f"Ma trận nhị phân thực tế:\n{y_true_bin}")
print("-" * 30)
print(f"Macro-Precision: {macro_p:.4f}")
print(f"Macro-Recall:    {macro_r:.4f}")
print(f"Macro-F1:        {macro_f1:.4f}")