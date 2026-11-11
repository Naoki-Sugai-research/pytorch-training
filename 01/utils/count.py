from utils.function import add


def count_word(text, target):

    # 第一引数がstr型か確認
    assert isinstance(text, str)

    # 第二引数がstr型かつ長さが1か確認
    assert isinstance(target, str) and len(target) == 1

    count = 0

    for char in text:
        if char == target:
            count = add(count, 1)

    return count