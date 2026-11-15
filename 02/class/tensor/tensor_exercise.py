import numpy as np
import torch


# =====================================
# 元データ：3クラス、各クラス5人、2科目
# =====================================

data = np.array([
    [[85, 78], [67, 82], [92, 88], [75, 70], [60, 64]],
    [[70, 68], [77, 72], [85, 90], [60, 65], [78, 76]],
    [[80, 84], [88, 87], [66, 68], [72, 73], [64, 60]]
])


# =====================================
# 問題1：NumPy配列をTensorに変換
# =====================================

tensor = torch.tensor(data, dtype=float)

print("【問題1：Tensorへの変換】")
print(tensor)
print("shape:", tensor.shape)


# =====================================
# 問題2：軸の並び替え
# 科目、クラス、人数の順番に変更
# =====================================

subjects_first = tensor.permute(2, 0, 1)

print("\n【問題2：軸の並び替え】")
print(subjects_first)
print("shape:", subjects_first.shape)


# =====================================
# 問題3：各生徒の2科目の合計点
# =====================================

student_totals = subjects_first.sum(dim=0)

print("\n【問題3：各生徒の合計点】")
print(student_totals)
print("shape:", student_totals.shape)


# =====================================
# 問題4：クラスごとの平均点
# =====================================

class_means = student_totals.mean(dim=1)

print("\n【問題4：クラスごとの平均点】")
print(class_means)


# =====================================
# 問題5：meanを使わずに平均点を求める
# =====================================

number_of_students = student_totals.shape[1]

class_means_without_mean = (
    student_totals.sum(dim=1) / number_of_students
)

print("\n【問題5：meanを使わない平均点】")
print(class_means_without_mean)


# =====================================
# 問題4と問題5の結果が一致するか確認
# =====================================

print("\n【結果の確認】")
print(torch.allclose(class_means, class_means_without_mean))