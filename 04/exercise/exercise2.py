from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


# Datasetクラスの作成
class MyDataset(Dataset):

    def __init__(self, dataset_dir):
        # データディレクトリの絶対パスを取得
        dir_path_resolved = Path(dataset_dir).resolve()

        # PNG画像のパス一覧を取得
        self.img_list = list(dir_path_resolved.glob("**/*.png"))

    # データセットの枚数を返す
    def __len__(self):
        return len(self.img_list)

    # 指定された番号の画像を取り出す
    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # 画像ファイルを読み込む
        img = Image.open(img_path)

        return img


if __name__ == "__main__":

    # Datasetのインスタンスを作成
    my_dataset = MyDataset("C:/Users/aaa26/pytorch-training/04/exercise/data")

    # 問題1.1：データセットの枚数を表示
    print("===== problem1.1 =====")
    print(len(my_dataset))

    # 問題1.2：画像サイズを表示
    print("===== problem1.2 =====")
    print(my_dataset[0].size)