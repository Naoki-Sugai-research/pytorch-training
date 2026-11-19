from pathlib import Path

# dataディレクトリのパスを指定
data_dir = Path("04/exercise/data")


# 問題1：dataディレクトリの絶対パスを表示
print("--- problem1 ---")
print(data_dir.resolve())


# 問題2：dataディレクトリ配下のディレクトリをすべて表示
print("--- problem2 ---")

dir_list = data_dir.glob("*")

for path in sorted(dir_list):
    if path.is_dir():
        print(path)


# 問題3：dataディレクトリ配下のPNG画像の数を表示
print("--- problem3 ---")

png_files = list(data_dir.glob("**/*.png"))

print(len(png_files))