import os
from typing import List

from tqdm import tqdm

import random
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from datasets import load_from_disk

from text_guided_cl.evaluate.metrics import compute_metrics
from text_guided_cl.evaluate.metrics import get_assignment
from text_guided_cl.evaluate.datasets import get_num_classes_dict


def read_values(data_dir, load_dir, pred_save_dir, datasets, embedding_type: str = "tfidf", dataset_info=None):
    full_metrics = {}

    for dset in datasets.items():
        print(dset)
        dataset = load_from_disk(os.path.join(data_dir, load_dir, dset))

        label_encoder = LabelEncoder().fit(dataset["cluster_label"])
        dataset = dataset.add_column("cluster_label_id", label_encoder.transform(dataset['cluster_label']))
        n_clusters = dataset_info[dset]

        metric_list = []
        prediction_dir = os.path.join(pred_save_dir, load_dir)
        for pred_file in os.listdir(prediction_dir):
            if pred_file.startswith(f"{dset}_{embedding_type}"):
                predictions = joblib.load(os.path.join(prediction_dir, pred_file))
                metrics = compute_metrics(predictions=predictions, labels=np.array(dataset["cluster_label_id"]))
                metric_list.append(metrics)

        full_metrics[dset] = metric_list[0]

    return full_metrics


def create_all_kmeans_predictions(load_dir, dataset, random_states=[42, 48, 112], caption_type: str = "all"):

    num_classes = get_num_classes_dict()

    dataset_predictions = []

    for random_state in tqdm(random_states):
        state_predictions = {}
        for embedding_type in ["visual", "tfidf", "sbert"]:
            n_clusters = num_classes[dataset]
            dset = load_from_disk(os.path.join(load_dir, dataset))

            label_encoder = LabelEncoder().fit(dset["cluster_label"])
            dset = dset.add_column("cluster_label_id", label_encoder.transform(dset['cluster_label']))

            if embedding_type == "tfidf":
                tfidf = TfidfVectorizer(
                    ngram_range=(1, 1),
                    max_features=6000,
                    stop_words='english'
                )
                concat_captions = ['. '.join(captions) for captions in dset["captions"]]
                if isinstance(caption_type, int):
                    concat_captions = [captions[caption_type] for captions in dset["captions"]]
                enc = tfidf.fit_transform(concat_captions)
            elif embedding_type == "sbert":
                enc = np.array(dset["sbert_embedding"])
                if isinstance(caption_type, int):
                    enc = np.array(dset[f"sbert_embedding_{caption_type}"])
            elif embedding_type == "visual":
                enc = np.array(dset["visual_embedding"]).squeeze()
            else:
                raise ValueError("embedding_type not defined")
            # text = dataset

            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1).fit(enc)

            labels = np.array(dset["cluster_label_id"])
            prediction = {
                "n_clusters": n_clusters,
                "random_state": random_state,
                "embedding_type": embedding_type,
                "load_dir": load_dir,
                "dset": dataset,
                "predictions": kmeans.labels_,
                "cluster_centers": kmeans.cluster_centers_,
                "inertia": kmeans.inertia_,
                "labels": labels,
                "metrics": compute_metrics(predictions=kmeans.labels_, labels=labels),
            }
            state_predictions[embedding_type] = prediction
        dataset_predictions.append(state_predictions)
    return dataset_predictions


def create_kmeans_prediction(data_dir, load_dir, pred_save_dir, datasets, embedding_type: str = "tfidf",
                             random_state=42):
    full_predictions = {}

    for dset, n_clusters in datasets.items():
        dataset = load_from_disk(os.path.join(data_dir, load_dir, dset))

        label_encoder = LabelEncoder().fit(dataset["cluster_label"])
        dataset = dataset.add_column("cluster_label_id", label_encoder.transform(dataset['cluster_label']))

        if embedding_type == "tfidf":
            tfidf = TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=6000,
                stop_words='english'
            )
            concat_captions = ['. '.join(captions) for captions in dataset["captions"]]
            enc = tfidf.fit_transform(concat_captions)
        elif embedding_type == "sbert":
            enc = np.array(dataset["sbert_embedding"])
        elif embedding_type == "visual":
            enc = np.array(dataset["visual_embedding"]).squeeze()
        else:
            raise ValueError("embedding_type not defined")
        # text = dataset

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1).fit(enc)

        predictions = {
            "n_clusters": n_clusters,
            "random_state": random_state,
            "embedding_type": embedding_type,
            "load_dir": load_dir,
            "dset": dset,
            "predictions": kmeans.labels_,
            "labels": np.array(dataset["cluster_label_id"]),
        }
        if pred_save_dir is not None:
            folder = os.path.join(pred_save_dir, load_dir)
            os.makedirs(folder, exist_ok=True)

            joblib.dump(predictions, os.path.join(folder, f"{dset}_{embedding_type}_{random_state}.pbz2"))

        # with tfidf

        full_predictions[dset] = predictions

    return full_predictions


def compute_metrics_per_prediction(predictions):
    metrics = compute_metrics(predictions=predictions["predictions"], labels=predictions["labels"])
    return metrics


def compute_metrics_all_predictions(prediction_dict):
    for load_dir in prediction_dict.keys():
        for embedding_type in prediction_dict[load_dir].keys():
            for i in range(len(prediction_dict[load_dir][embedding_type])):
                for dset in prediction_dict[load_dir][embedding_type][i].keys():
                    prediction_dict[load_dir][embedding_type][i][dset]["metrics"] = compute_metrics_per_prediction(
                        prediction_dict[load_dir][embedding_type][i][dset]
                    )
    return prediction_dict


def aggregate_random_state_metrics(prediction_dict):
    aggregated_dict = {}
    for load_dir in prediction_dict.keys():
        aggregated_dict[load_dir] = {}
        for embedding_type in prediction_dict[load_dir].keys():
            aggregated_dict[load_dir][embedding_type] = {}
            for dset in prediction_dict[load_dir][embedding_type][0].keys():
                aggregated_dict[load_dir][embedding_type][dset] = {}
                aggregated_metrics = {}
                for metric in prediction_dict[load_dir][embedding_type][0][dset]["metrics"].keys():
                    aggregated_metrics[metric] = []
                for i in range(len(prediction_dict[load_dir][embedding_type])):
                    for metric in prediction_dict[load_dir][embedding_type][i][dset]["metrics"].keys():
                        aggregated_metrics[metric].append(
                            prediction_dict[load_dir][embedding_type][i][dset]["metrics"][metric])
                for metric in aggregated_metrics.keys():
                    aggregated_metrics[metric] = np.mean(aggregated_metrics[metric])
                aggregated_dict[load_dir][embedding_type][dset] = aggregated_metrics

    return aggregated_dict


def get_metric_df_by_load_dir(aggregated_dict, load_dir, metric="accuracy"):
    df = []
    dsets = []
    for embedding_type in aggregated_dict[load_dir].keys():
        row = [load_dir, embedding_type]
        for dset in aggregated_dict[load_dir][embedding_type].keys():
            # if dset == "dtd":
            #     continue
            if dset not in dsets:
                dsets.append(dset)
            row.append(aggregated_dict[load_dir][embedding_type][dset][metric])
        df.append(row)
    columns = ["load_dir", "embedding_type"]
    df = pd.DataFrame(df, columns=columns + dsets)
    df[dsets] = df[dsets].round(3)
    df["avg"] = df[dsets].mean(axis=1).round(3)
    return df


def create_all_predictions(
        data_dir: str, load_dirs: List[str], pred_save_dir: str = None, datasets=None, num_random_states=3
):
    assert datasets is not None

    all_predictions = {}
    embedding_types = ["tfidf", "sbert", "visual"]
    j = 1
    for load_dir in load_dirs:
        all_predictions[load_dir] = {}
        for embedding_type in embedding_types:
            random_state_predictions = []
            for i in range(num_random_states):
                print(f"Run prediction {j} of {len(load_dirs) * len(embedding_types) * num_random_states}")
                j += 1

                random_state = random.randint(1, 100)
                random_state_predictions.append(
                    create_kmeans_prediction(
                        data_dir=data_dir,
                        load_dir=load_dir,
                        pred_save_dir=pred_save_dir,
                        datasets=datasets,
                        embedding_type=embedding_type,
                        random_state=random_state
                    )
                )
            all_predictions[load_dir][embedding_type] = random_state_predictions

    if pred_save_dir is not None:
        joblib.dump(all_predictions, os.path.join(pred_save_dir, "all_predictions.pbz2"))
    return all_predictions


# create all predictions with hierarchical stuff (new_2 and new_3 dsets)
# compute metrics, average, show as Pandas
# create df such that I compare flamingo vs blip vs vqa
# compute assignment, then compute confusion matrix

def get_confusion_matrix(prediction_dict):
    # assgn is 2D numpy array with (n_clusters, assignment_class)
    assgn = get_assignment(true_row_labels=prediction_dict["labels"],
                           predicted_row_labels=prediction_dict["predictions"])
    translation_dict = {}
    for i in range(assgn.shape[0]):
        translation_dict[assgn[i, 1]] = assgn[i, 0]

    new_preds = np.zeros(prediction_dict["predictions"].shape)
    for i in range(len(new_preds)):
        new_preds[i] = translation_dict[prediction_dict["predictions"][i]]
    return confusion_matrix(y_true=prediction_dict["labels"], y_pred=new_preds)


def get_label_dict(dataset):
    label_dict = {}

    def fill_dict(sample):
        if sample["cluster_label"] not in label_dict:
            label_dict[sample["cluster_label"]] = sample["label"]
        else:
            assert label_dict[sample["cluster_label"]] == sample["label"]

    dataset.map(fill_dict)

    label_list = []
    for i in range(len(label_dict.keys())):
        label_list.append(label_dict[i])

    return label_dict, label_list


def get_averaged_confusion_mtrix(full_prediction_dict, load_dir, embedding_type, dataset):
    prediction_dict = full_prediction_dict[load_dir][embedding_type]

    full_cm = get_confusion_matrix(prediction_dict[0][dataset])
    for i in range(1, len(prediction_dict)):
        full_cm += get_confusion_matrix(prediction_dict[i][dataset])

    full_cm = np.rint(full_cm / len(prediction_dict)).astype(int)

    return full_cm


def get_cm_display(full_prediction_dict, data_dir, load_dir, embedding_type, dataset_name):
    dataset = load_from_disk(os.path.join(data_dir, load_dir, dataset_name))
    label_dict, label_list = get_label_dict(dataset)

    full_cm = get_averaged_confusion_mtrix(full_prediction_dict, load_dir, embedding_type, dataset_name)
    return ConfusionMatrixDisplay(full_cm, display_labels=label_list)
