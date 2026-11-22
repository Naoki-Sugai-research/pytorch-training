import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


if __name__ == "__main__":

    # 画像ファイルの読み込み
    image_path = Path(__file__).resolve().parent / "exercise_data" / "dog_img.png"
    image = Image.open(image_path).convert("RGB")

    # Normalizeの設定
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # --------------------------------
    # 1. 画像のリサイズと正規化
    # --------------------------------
    preprocess_1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --------------------------------
    # 2. ランダムな水平反転と色調変換
    # --------------------------------
    preprocess_2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
        ),
        transforms.ToTensor()
    ])

    # --------------------------------
    # 3. ランダムな回転とクロップ
    # --------------------------------
    preprocess_3 = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor()
    ])

    # それぞれの前処理をリストにまとめる
    preprocess_list = [
        preprocess_1,
        preprocess_2,
        preprocess_3
    ]

    # 各前処理を実行して画像を表示
    for i, preprocess in enumerate(preprocess_list, start=1):

        processed_image = preprocess(image)

        # Tensorの軸を(C,H,W)から(H,W,C)に変更
        image_array = processed_image.permute(1, 2, 0).numpy()

        # 画像の表示
        plt.figure(figsize=(5, 5))
        plt.imshow(image_array)
        plt.title(f"Preprocessing {i}")
        plt.axis("off")
        plt.show()
