import re


def process_train_data(file_name="人民日报标记语料库及部分验证数据示例/199801.txt"):
    with open(file_name, 'r', encoding='gbk') as f:
        text = f.read()
    # 处理[
    text = re.sub("\[", "[/w  ", text)
    # 处理]
    text = re.sub("]\w+ ", u"  ]/w ", text)
    sentences = text.split('\n')
    # 去除前面的日期信息(如果不去除，第一个位置的数是m的比重会非常大，不利用模型识别其他类型数据)
    sentences = [sentence[23:] for sentence in sentences]
    text = "\n".join(sentences)

    file_name = "train/processed_text.txt"
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)


def process_test_data(file_name="人民日报标记语料库及部分验证数据示例/供测试的验证预料示例.txt"):
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    file_name = "test/processed_text.txt"
    # 去除空格
    text = re.sub(" ", "", text)
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == "__main__":
    process_train_data()
    process_test_data()
