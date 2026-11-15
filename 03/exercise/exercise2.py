
import torch
from torch import nn


if __name__ == "__main__":

    # =====================================
    # 問題1：入力テンソルの作成
    # =====================================
    print("===problem1===")

    _in = torch.ones((32, 1024))

    print("_in :", _in.shape)


    # =====================================
    # 問題2：出力を32×256にする
    # =====================================
    print("===problem2===")

    fc1 = nn.Linear(
        in_features=1024,
        out_features=256
    )

    out1 = fc1(_in)

    print("out1 :", out1.shape)


    # =====================================
    # 問題3：出力を32×2048にする
    # =====================================
    print("===problem3===")

    fc2 = nn.Linear(
        in_features=1024,
        out_features=2048
    )

    out2 = fc2(_in)

    print("out2 :", out2.shape)


    # =====================================
    # Appendix：32×16×16に形状変更
    # =====================================
    print("===appendix===")

    out3 = out1.view(32, 16, 16)

    print("out3 :", out3.shape)
