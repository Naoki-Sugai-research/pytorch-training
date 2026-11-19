from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset


# Datasetクラスの作成
class MyDataset(Dataset):

    def __init__(self, dataset_dir):

        # データディレクトリの絶対パスを取得
        dir_path_resolved = Path(dataset_dir).resolve()

        # 配下にあるPNG画像のパス一覧を取得
        self.img_list = list(dir_path_resolved.glob("**/*.png"))

        # 画像をTensor型に変換する処理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # データセットの枚数を返す
    def __len__(self):
        return len(self.img_list)

    # 指定された番号の画像とラベルを取得
    def __getitem__(self, idx):

        # 画像ファイルのパスを取得
        img_path = self.img_list[idx]

        # 画像を読み込む
        img = Image.open(img_path)

        # Tensor型に変換
        img_tensor = self.transform(img)

        # パスを構成要素に分解
        parts = img_path.parts

        # 親ディレクトリ名をラベルにする
        label = parts[-2]

        # 画像とラベルをセットで返す
        return img_tensor, label


if __name__ == "__main__":

    # Datasetのインスタンスを作成
    data_dir = Path(__file__).resolve().parent / "data"
    dataset = MyDataset(data_dir)

    # 問題1：画像サイズを表示
    print("---problem1---")

    img, label = dataset[0]

    print(f"img_size : {img.shape}")

    # 問題2：ラベルを表示
    print("---problem2---")

    print(f"label : {label}")