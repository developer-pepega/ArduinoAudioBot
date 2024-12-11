from natasha import Doc
from natasha import Segmenter
from natasha import NewsEmbedding
from natasha import NewsMorphTagger
from natasha import NewsSyntaxParser
from natasha import NewsNERTagger

number_dict = {
    "ноль": 0,"один": 1,"два": 2,"три": 3,"четыре": 4,"пять": 5,
    "шесть": 6,"семь": 7,"восемь": 8,"девять": 9,"десять": 10,"одиннадцать": 11,
    "двенадцать": 12,"тринадцать": 13,"четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16,"семнадцать": 17,"восемнадцать": 18,"девятнадцать": 19,"двадцать": 20,
    "тридцать": 30,"сорок": 40,"пятьдесят": 50,"шестьдесят": 60,
    "семьдесят": 70,"восемьдесят": 80,"девяносто": 90,"сто": 100,"двести": 200,"триста": 300,"четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700,"восемьсот": 800,
    "девятьсот": 900, "тысяча": 1000, "миллион": 1000000, "миллиард": 1000000000
}

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

def translator_text_to_num(text: str):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    return sum([number_dict[t.text] for t in doc.tokens if t.pos == "NUM"])

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[m][n]

def find_closest_command_index(word, commands):
    min_distance = float('inf')
    closest_index = (-1, -1)
    max_length = max(len(word), max(len(cmd) for sublist in commands for cmd in sublist))

    for i, sublist in enumerate(commands):
        for j, command in enumerate(sublist):
            distance = levenshtein_distance(word, command)
            if distance < min_distance:
                min_distance = distance
                closest_index = (i, j)

    confidence = 1 - (min_distance / max_length)
    return closest_index, confidence

import re
def preprocess_text(text):
    """
    Переводит в строку в нижний регистр и убирает знаки препинания.

    :param text: Исходная строка
    :return: Строка в нижним регистре без знаков препинания
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def clean_string_until_word(input_string, word):
    """
    Очищает строку до указанного слова.

    :param input_string: Исходная строка
    :param word: Слово, до которого нужно очистить строку
    :return: Очищенная строка
    """
    index = input_string.find(word)

    if index != -1:
        return input_string[index:].strip()
    else:
        return 0
    
commands = [
    ["вперед","двигайся вперед","иди вперед","продолжай движение","двигайся прямо","шагай вперед","продвигайся вперед","следуй вперед","двигайся дальше","иди прямо"],
    ["поверни налево","иди налево","двигайся влево","поворот налево","налево","сверни налево","двигайся налево","иди влево","поверни влево","влево"],
    ["поверни направо","иди направо","двигайся вправо","поворот направо","направо","сверни направо","двигайся направо","иди вправо","поверни вправо","вправо"],
    ["остановись","стой","прекрати движение","замри","остановка","останови движение","прекрати идти","прекрати двигаться","хватит","перестань ехать"],
    ["назад","двигайся назад","иди назад","отступи назад","вернись назад","двигайся обратно","иди обратно","отступи","вернись","двигайся назад"]
]

def translator_text_to_cmd(data):
    """
    Обработка текста в команду

    :param data: Строка
    :return: Команду и угол поворота
    """
    text = preprocess_text(data)
    text = clean_string_until_word(text,"паша")
    if text == 0:
        return 7, 0
        
    text_for_comand = " ".join(text.split()[1:3])
    comand_list = find_closest_command_index(text_for_comand,commands)
    if comand_list[1] < 0.8:
        return 7, 0
    if comand_list[0][0] == 1 or comand_list[0][0] == 2:
        degree = translator_text_to_num(text)
    else:
        degree = 0
    return comand_list[0][0], degree