import spacy
from collections import Counter
from string import punctuation

nlp = spacy.load("en_core_web_sm")


def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result


def get_most_common_words(text, num_words):
    output = get_hotwords(text)
    most_common_list = Counter(output).most_common(num_words + 5)
    most_common_list = [a[0] for a in most_common_list[0:num_words]]
    return most_common_list


def add_most_common_words(sample):
    sample["keywords"] = get_most_common_words(' , '.join(sample["captions"]), 3)
    return sample
