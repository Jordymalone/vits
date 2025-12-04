from viphoneme import vi2IPA
VI_CODE = "6"

def process_string(input_str):
    # print("input:",input_str)
    # 將出現的 .. 處理成單個 .
    input_str = input_str.replace("..", ".")
    # 不需要斷詞
    input_str = input_str.replace("_", " ")

    input_str = add_language_code(input_str)

    return input_str

def process_vietnamese_sentence(sentence):
    # print(vi2IPA(sentence))
    trn2WithoutHypen = process_string(vi2IPA(sentence))
    # print(trn2WithoutHypen)
    return trn2WithoutHypen

def add_language_code(sentence):
    origin_word_list = sentence.split()
    for index, word in enumerate(origin_word_list):
        if (word != "."):
            origin_word_list[index] = word[:-1] + VI_CODE + word[-1:]
    return " ".join(origin_word_list)
    


if __name__ == "__main__":
    testCase = "chúng tôi ngồi ăn uống trò chuyện vui vẻ với nhau"
#    testCase = "đám đông bu vào đôgn nghẹt cố chen lên trên"
    while True:
        testCase = input('enter sentence:')
        print(process_vietnamese_sentence(testCase))

