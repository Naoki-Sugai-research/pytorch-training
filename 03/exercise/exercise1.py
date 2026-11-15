
import torch
from torch import nn


if __name__ == "__main__":

    # =====================================
    # 問題1：テンソルの作成
    # =====================================
    print("===problem1===")

    my_tensor = torch.ones((32, 3, 128, 128))

    print("original :", my_tensor.shape)


    # =====================================
    # 問題2：kernel_size = 3
    # 出力：32 × 64 × 126 × 126
    # =====================================
    print("===problem2===")

    conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3
    )

    with torch.no_grad():
        out1 = conv1(my_tensor)

    print("out1 :", out1.shape)


    # =====================================
    # 問題3：kernel_size = 3
    # 出力：32 × 256 × 64 × 64
    # =====================================
    print("===problem3===")

    conv2 = nn.Conv2d(
        in_channels=3,
        out_channels=256,
        kernel_size=3,
        stride=2,
        padding=1
    )

    with torch.no_grad():
        out2 = conv2(my_tensor)

    print("out2 :", out2.shape)


    # =====================================
    # 問題4：kernel_size = 5
    # =====================================
    print("===problem4===")

    # 問題2と同じ出力サイズ
    conv3 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=5,
        stride=1,
        padding=1
    )

    # 問題3と同じ出力サイズ
    conv4 = nn.Conv2d(
        in_channels=3,
        out_channels=256,
        kernel_size=5,
        stride=2,
        padding=2
    )

    with torch.no_grad():
        out3 = conv3(my_tensor)
        out4 = conv4(my_tensor)

    print("out1(kernel=5) :", out3.shape)
    print("out2(kernel=5) :", out4.shape)
