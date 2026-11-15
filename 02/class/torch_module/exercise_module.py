import torch
from torch import nn


# =====================================
# MyModelクラスの作成
# =====================================

class MyModel(nn.Module):

    def __init__(
        self,
        mytensor,
        elem_add,
        elem_multiply
    ):

        # 親クラス nn.Module の初期化
        super().__init__()

        # 計算に使用する値をインスタンス変数に保存
        self.mytensor = mytensor
        self.elem_add = elem_add
        self.elem_multiply = elem_multiply


    def forward(self, x):

        # 問題2：入力にmytensorを加算
        result2 = x + self.mytensor

        # 問題3：テンソル全体にelem_addを加算
        result3 = result2 + self.elem_add

        # 問題4：テンソル全体にelem_multiplyを乗算
        result4 = result3 * self.elem_multiply

        # 各段階の結果を返す
        return result2, result3, result4


# =====================================
# 入力データの準備
# =====================================

# 3行3列、すべて1.0の入力Tensor
input_tensor = torch.ones((3, 3))

# 3行3列、すべて2.0のTensor
mytensor = torch.full(
    (3, 3),
    2.0
)


# =====================================
# モデルの作成
# =====================================

model = MyModel(
    mytensor=mytensor,
    elem_add=4.0,
    elem_multiply=6.0
)


# =====================================
# モデルの実行
# =====================================

result2, result3, result4 = model(input_tensor)


# =====================================
# 結果の出力
# =====================================

print("===== 問題2 =====")
print(result2)

print("===== 問題3 =====")
print(result3)

print("===== 問題4 =====")
print(result4)