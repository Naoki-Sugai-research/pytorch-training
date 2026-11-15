
import torch
from torch import nn


# モデルクラスの定義
class ExerciseModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ① Convolution層
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=256,
            kernel_size=5,
            stride=8,
            padding=2
        )

        # ② Batch Normalization層
        self.bn = nn.BatchNorm2d(
            num_features=256
        )

        # ③ ReLU層
        self.relu = nn.ReLU()

        # ④ Linear層
        self.fc = nn.Linear(
            in_features=256 * 16 * 16,
            out_features=64
        )

    # 入力から出力までの処理
    def forward(self, x):

        # Convolution
        x = self.conv(x)

        # Batch Normalization
        x = self.bn(x)

        # ReLU
        x = self.relu(x)

        # Tensorの形状を変更
        x = torch.flatten(x, start_dim=1)

        # Linear
        x = self.fc(x)

        return x


# プログラムの実行部分
if __name__ == "__main__":

    # 入力Tensorの定義
    in_tensor = torch.ones(32, 3, 128, 128)

    # モデルインスタンスの作成
    model = ExerciseModel()

    # モデルの実行
    with torch.no_grad():
        out = model(in_tensor)

    # 結果の確認
    print(f"in  : {in_tensor.shape}")
    print(f"out : {out.shape}")
