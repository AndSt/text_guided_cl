from tqdm import tqdm

import pandas as pd

from gensim.summarization import keywords
from keybert import KeyBERT
import openai

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import spacy
from sentence_transformers import SentenceTransformer


def get_gensim_keywords(df, label_names, label_column: str = "label", samples_per_cluster: int = 2000, top_n: int = 5):
    preds_df = []
    for i in tqdm(range(len(label_names))):
        d2 = df[df[label_column] == i]
        text = " ".join(d2.sample(min(len(d2), samples_per_cluster))["text"].tolist())

        kwds = keywords(text, words=top_n, lemmatize=True, split=True)
        preds_df.append([label_names[i], kwds])
    preds_df = pd.DataFrame(preds_df, columns=["truth", "pred"])
    return preds_df


# the for loop before as a function
def get_keybert_keywords(df, label_names, label_column: str = "label", samples_per_cluster: int = 2000, top_n: int = 5):
    preds_df = []
    for i in tqdm(range(len(label_names))):
        d2 = df[df[label_column] == i]
        text = " ".join(d2.sample(min(len(d2), samples_per_cluster))["text"].tolist())

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, top_n=top_n)
        preds_df.append([label_names[i], ", ".join([kw[0] for kw in keywords])])
    preds_df = pd.DataFrame(preds_df, columns=["truth", "pred"])
    return preds_df


def chat_call(content: str, model: str = "gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return completion


def get_chatgpt_keywords(df, label_names, label_column: str = "label", samples_per_cluster: int = 2000, top_n: int = 5):
    kws = {}
    for i in tqdm(range(len(label_names))):
        d2 = df[df[label_column] == i]
        text = " ".join(d2.sample(min(len(d2), samples_per_cluster))["text"].tolist())

        content = f"""{text}

Provide {top_n} keywords describing the previous text in a comma-separated list:
"""
        kws[i] = chat_call(content)
    pred_df = pd.DataFrame({"truth": label_names, "pred": [v.choices[0].message.content for k, v in kws.items()]})
    return pred_df


def split(text, splitters=[",", "."]):
    split_text = [text]
    for splitter in splitters:
        new_text = []
        for t in split_text:
            new_text.extend(t.split(splitter))
        split_text = new_text
    return split_text


def get_statistics_keywords(df, label_names, label_column: str = "label", top_n: int = 5):
    full_kws = {}
    for i in range(len(label_names)):
        class_df = df[df[label_column] == i]
        kws = [w.strip() for s in [split(t, splitters=[",", "."]) for t in class_df["text"].tolist()] for w in s]
        full_kws[label_names[i]] = pd.Series(kws).value_counts().index[0:200].tolist()

    current_kws = {k: v[0:top_n] for k, v in full_kws.items()}
    other_kws = {k: v[top_n:] for k, v in full_kws.items()}

    def find_duplicate_and_remove(current_kws, other_kws):
        is_intersection = False
        for k, v in current_kws.items():
            for k2, v2 in current_kws.items():
                if k == k2:
                    continue
                intersection = list(set(v).intersection(set(v2)))
                if len(intersection) > 0:
                    is_intersection = True
                    for i in intersection:
                        for k3, v3 in current_kws.items():
                            if i in v3:
                                current_kws[k3].remove(i)
                        for k3, v3 in other_kws.items():
                            if i in v3:
                                other_kws[k3].remove(i)
        return is_intersection, current_kws, other_kws

    def fill_up_values(current_kws, other_kws, length=2):
        for k, v in current_kws.items():
            if len(v) < length:
                # print(len(v), length-len(v), len(other_kws[k]), len(current_kws[k]))
                current_kws[k].extend(other_kws[k][0:length - len(v)])
                other_kws[k] = other_kws[k][length - len(v) + 1:]
                # print(len(v), length-len(v), len(other_kws[k]), len(current_kws[k]))
        return current_kws, other_kws

    z = 0
    while True:
        is_intersection, current_kws, other_kws = find_duplicate_and_remove(current_kws, other_kws)
        if not is_intersection:
            break
        current_kws, other_kws = fill_up_values(current_kws, other_kws, length=top_n)
        z += 1

    preds_df = []
    for k, v in current_kws.items():
        preds_df.append([k, ", ".join(v)])
    preds_df = pd.DataFrame(preds_df, columns=["truth", "pred"])
    return preds_df


def string_match(pred_df, exact_match=True):
    def check_match(row):
        truth_labels = row["truth"].lower().replace("_", " ")
        truth_labels = split(truth_labels, splitters=[",", "."])
        truth_labels = [x.strip().lower() for x in truth_labels]

        pred_labels = row["pred"].lower().strip()

        for j in range(len(truth_labels)):
            if exact_match:
                if truth_labels[j] == pred_labels:
                    return 1
            else:
                if truth_labels[j] in pred_labels:# or pred_labels in truth_labels[j]:
                    return 1
            # print(pred_labels[j], truth_label)
        return 0

    pred_df["match"] = pred_df.apply(check_match, axis=1)
    return pred_df

nlp = spacy.load('en_core_web_md')

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



def get_similarity(df, similarity_threshold=0.4):
    def check_similarity(row):
        pred_labels = split(row["pred"], splitters=[",", "."])
        pred_labels = [x.strip().lower() for x in pred_labels]

        truth_label = row["truth"].lower()
        truth_label = truth_label.replace("_", " ")

        sims = []
        for j in range(len(pred_labels)):
            sims.append(nlp(pred_labels[j]).similarity(nlp(truth_label)))

        return sims

    df["similarity"] = df.apply(check_similarity, axis=1)
    df["match"] = df["similarity"].apply(lambda x: np.any(np.array(x) > similarity_threshold))
    return df


def get_sbert_similarity(df, similarity_threshold=0.4):
    def check_similarity(row):
        pred_labels = row["pred"].strip().lower()

        truth_label = row["truth"].lower()
        truth_label = truth_label.replace("_", " ")

        sims = [cosine_similarity(sbert_model.encode([pred_labels]), sbert_model.encode([truth_label]))[0][0]]
        return sims

    df["similarity"] = df.apply(check_similarity, axis=1)
    df["match"] = df["similarity"].apply(lambda x: np.any(np.array(x) > similarity_threshold))
    return df