import os

import pandas as pd
import numpy as np

import joblib

from text_guided_cl.evaluate.metrics import get_assignment
from datasets import load_from_disk

from sklearn.preprocessing import LabelEncoder


def load_ground_truth(data_dir: str, load_dir: str, dataset_name: str, label_col: str = "cluster_label"):
    analyze_dir = os.path.join(data_dir, load_dir)
    analyzed_dir = os.path.join(data_dir, f"analyzed_{load_dir}")

    dataset = load_from_disk(os.path.join(analyze_dir, dataset_name))
    le = LabelEncoder().fit(dataset[label_col])
    labels = le.transform(dataset[label_col])
    text = [". ".join(sample["captions"]) for sample in dataset]

    columns = ["text", "label"]
    df = pd.DataFrame(list(zip(text, labels)), columns=columns)

    metrics = joblib.load(os.path.join(analyzed_dir, dataset_name, "metrics.pbz2"))
    prediction_dict = metrics[0]["sbert"]
    assgn = get_assignment(true_row_labels=prediction_dict["labels"],
                           predicted_row_labels=prediction_dict["predictions"])

    translation_dict = {}
    for i in range(assgn.shape[0]):
        translation_dict[assgn[i, 1]] = assgn[i, 0]

    new_preds = np.zeros(prediction_dict["predictions"].shape)
    for i in range(len(new_preds)):
        new_preds[i] = translation_dict[prediction_dict["predictions"][i]]
    df["predictions"] = new_preds.astype(int)

    return df, le.classes_
