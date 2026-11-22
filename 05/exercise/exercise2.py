
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


# =====================================
# 画像の前処理を行うクラス
# =====================================
class ImageTransform():

    def __init__(self, resize, mean, std):

        self.data_transform = {

            # 学習用の前処理
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            # 評価用の前処理
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


# =====================================
# Datasetを作成するクラス
# =====================================
class MyDataset(Dataset):

    def __init__(self, img_list, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        # 画像ファイルのパスを取得
        img_path = self.img_list[idx]

        # 画像を読み込む
        img = Image.open(img_path).convert("RGB")

        # 画像の前処理
        img_tensor = self.transform(img, self.phase)

        # 親フォルダ名からラベルを取得
        img_path = Path(img_path)
        parts = img_path.parts
        label = int(parts[-2])

        return img_tensor, label


# =====================================
# メイン処理
# =====================================
if __name__ == "__main__":

    # ---------------------------------
    # 画像ファイルのパス一覧を作成
    # ---------------------------------

    project_dir = Path(__file__).resolve().parents[2]

    data_dir = project_dir / "04" / "exercise" / "data"

    file_list = sorted(data_dir.glob("*/*.png"))

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"データフォルダが見つかりません: {data_dir}"
        )

    if not file_list:
        raise FileNotFoundError(
            f"画像ファイルが見つかりません: {data_dir}"
        )

    # ---------------------------------
    # 学習用と評価用にデータを分割
    # ---------------------------------

    dataset_size = len(file_list)

    train_ratio = 0.8
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_paths, val_paths = random_split(
        file_list,
        [train_size, val_size]
    )

    # ---------------------------------
    # 画像の前処理を設定
    # ---------------------------------

    size = 24
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # ---------------------------------
    # Datasetオブジェクトを作成
    # ---------------------------------

    train_dataset = MyDataset(
        train_paths,
        transform=ImageTransform(size, mean, std),
        phase='train'
    )

    val_dataset = MyDataset(
        val_paths,
        transform=ImageTransform(size, mean, std),
        phase='val'
    )

    # ---------------------------------
    # 演習2：データセットの要素数を出力
    # ---------------------------------

    print("===== problem2 =====")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"val dataset size: {len(val_dataset)}")

    # ---------------------------------
    # 演習3：DataLoaderを作成
    # ---------------------------------

    # 学習用DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    # 評価用DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    # ---------------------------------
    # 学習用DataLoaderの出力
    # ---------------------------------

    print("===== problem3: train =====")

    for batch in train_dataloader:

        # batchはデータとラベルのタプル
        data, labels = batch

        print("Data shape:", data.shape)
        print("Labels:", labels)

    # ---------------------------------
    # 評価用DataLoaderの出力
    # ---------------------------------

    print("===== problem3: val =====")

    for batch in val_dataloader:

        data, labels = batch

        print("Data shape:", data.shape)
        print("Labels:", labels)
