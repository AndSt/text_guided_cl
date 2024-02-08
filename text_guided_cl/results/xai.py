import os
import json
import joblib

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder

from text_guided_cl.evaluate.metrics import compute_metrics, get_assignment


def load_data_for_xai(base_data, analyze_dir, dataset_type, dataset_name, num_captions=4, repr="tfidf"):
    load_dir = os.path.join(analyze_dir, dataset_type, dataset_name)
    all_kmeans = joblib.load(os.path.join(load_dir, "all_kmeans.pbz2"))

    # open sub_samples.json
    with open(os.path.join(load_dir, "sub_samples.json"), "r") as f:
        sub_samples = json.load(f)

    samples_per_num_captions = len(sub_samples["1"])

    idx = (num_captions - 1) * samples_per_num_captions
    best_run = -1
    best_random_state = -1
    best_inertia = 0

    for run_id, run in enumerate(all_kmeans[idx]):
        run = run[repr]
        if run["inertia"] < best_inertia or best_random_state == -1:
            best_random_state = run["random_state"]
            best_inertia = run["inertia"]
            best_run = run_id

    prediction_dict = all_kmeans[idx][best_run][repr]

    dataset = load_from_disk(os.path.join(base_data, dataset_type, dataset_name))
    le = LabelEncoder().fit(dataset["label"])
    labels = le.transform(dataset["label"])
    label_names = le.classes_

    assgn = get_assignment(true_row_labels=labels, predicted_row_labels=prediction_dict["predictions"])

    translation_dict = {}
    for i in range(assgn.shape[0]):
        translation_dict[assgn[i, 1]] = assgn[i, 0]

    new_preds = np.zeros(prediction_dict["predictions"].shape)
    for i in range(len(new_preds)):
        new_preds[i] = translation_dict[prediction_dict["predictions"][i]]

    full_concatenated_captions = joblib.load(os.path.join(load_dir, "full_concatenated_captions.pbz2"))
    captions = [c[idx] for c in full_concatenated_captions]

    df = pd.DataFrame({"text": captions, "label": labels, "predictions": new_preds.astype(int)})

    # check if everything works well
    metr1 = compute_metrics(labels=labels, predictions=prediction_dict["predictions"])
    metr2 = compute_metrics(labels=prediction_dict["labels"], predictions=prediction_dict["predictions"])

    assert_almost_equal(np.array(list(metr1.values())), np.array(list(metr2.values())))

    if repr == "tfidf":
        other_metrics = all_kmeans[idx][best_run]["sbert"]["metrics"]
    else:
        other_metrics = all_kmeans[idx][best_run]["tfidf"]["metrics"]

    return df, label_names, prediction_dict['metrics'], other_metrics
