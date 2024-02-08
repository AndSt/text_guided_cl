import argparse
import os
import logging

from slurm_utils.convenience.flags import flags, FLAGS
from slurm_utils.convenience.log import init_logging
from absl import app

import joblib
import json

import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer
from datasets import load_from_disk

import random

from text_guided_cl.evaluate.metrics import compute_metrics
from text_guided_cl.evaluate.datasets import get_num_classes_dict

flags.DEFINE_string("load_dir", default="blip2_full", help="Number of keyword generations per cluster")
flags.DEFINE_string("save_dir", default="blip2_full_num_captions", help="Number of keyword generations per cluster")

flags.DEFINE_string("dataset", default="no_captions_caption", help="Type and name of the dataset")

flags.DEFINE_integer("num_test_captions", default=15, help="Number of keyword generations per cluster")
flags.DEFINE_integer("samplings_per_amount", default=8, help="Number of keyword generations per cluster")
flags.DEFINE_integer("num_random_states", default=50, help="Number of keyword generations per cluster")


def is_list_in_list_of_lists(lst, lst_of_lst):
    for sub_lst in lst_of_lst:
        if lst == sub_lst:
            return True
    return False


def generate_test_id_dict(id_list, num_test_captions, samplings_per_amount: int):
    sub_samples = {}
    for i in range(1, num_test_captions + 1):
        sub_samples[i] = []
        for j in range(samplings_per_amount):
            new_list = np.sort(np.random.choice(id_list, i, replace=False)).tolist()
            while is_list_in_list_of_lists(new_list, sub_samples[i]):
                new_list = np.sort(np.random.choice(id_list, i, replace=False)).tolist()
            sub_samples[i].append(new_list)

    return sub_samples


def get_embeddings_and_captions(model, dataset, sub_samples):
    full_embeddings = []
    full_concatenated_captions = []
    for i in tqdm(range(len(dataset))):
        sample_captions = dataset[i]["captions"]

        concatenated_captions = {}
        for key, value in sub_samples.items():
            concatenated_captions[key] = []
            for sample in value:
                concatenated_captions[key].append(". ".join([sample_captions[i] for i in sample]))

        concatenated_captions = [caption for captions in concatenated_captions.values() for caption in captions]
        full_concatenated_captions.append(concatenated_captions)
        embeddings = model.encode(concatenated_captions)
        full_embeddings.append(embeddings)
    full_np_embeddings = np.array(full_embeddings)
    full_np_embeddings = np.transpose(full_np_embeddings, (1, 0, 2))
    return full_np_embeddings, full_concatenated_captions


def embeddings_to_predictions(dataset, dataset_name, full_np_embeddings, full_concatenated_captions, random_states):
    label_encoder = LabelEncoder().fit(dataset["cluster_label"])
    labels = label_encoder.transform(dataset['cluster_label'])

    num_classes = get_num_classes_dict()
    n_clusters = num_classes[dataset_name]

    all_kmeans = {}

    num_runs = 0
    for single_subset_id in range(full_np_embeddings.shape[0]):
        random_state_predictions = []
        for random_state in random_states:
            state_predictions = {}
            for embedding_type in ["tfidf", "sbert"]:
                num_runs += 1

    print(f"Running {num_runs} runs")

    with tqdm(total=num_runs) as pbar:
        for single_subset_id in range(full_np_embeddings.shape[0]):
            random_state_predictions = []
            for random_state in random_states:
                state_predictions = {}
                for embedding_type in ["tfidf", "sbert"]:

                    if embedding_type == "tfidf":
                        tfidf = TfidfVectorizer(
                            ngram_range=(1, 1),
                            max_features=2000,
                            stop_words='english'
                        )
                        enc = tfidf.fit_transform([caption[single_subset_id] for caption in full_concatenated_captions])
                    elif embedding_type == "sbert":
                        enc = full_np_embeddings[single_subset_id]

                    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, init="k-means++").fit(enc)

                    prediction = {
                        "n_clusters": n_clusters,
                        "random_state": random_state,
                        "embedding_type": embedding_type,
                        "predictions": kmeans.labels_,
                        "cluster_centers": kmeans.cluster_centers_,
                        "inertia": kmeans.inertia_,
                        "labels": labels,
                        "metrics": compute_metrics(predictions=kmeans.labels_, labels=labels),
                    }
                    state_predictions[embedding_type] = prediction

                    pbar.update(1)

                random_state_predictions.append(state_predictions)
            all_kmeans[single_subset_id] = random_state_predictions

    return all_kmeans


def aggregate_metrics(all_kmeans):
    aggregated_metrics = []

    for k in all_kmeans.keys():
        aggregated = {}
        for j in all_kmeans[k][0].keys():  # tfidf, sbert
            aggregated[j] = {}
            for repr in all_kmeans[k][0][j]["metrics"].keys():
                for num_caps in range(len(all_kmeans[k])):  # random state
                    if repr not in aggregated[j].keys():
                        aggregated[j][repr] = []
                    aggregated[j][repr].append(all_kmeans[k][num_caps][j]["metrics"][repr])
                aggregated[j][repr] = [sum(aggregated[j][repr]) / len(aggregated[j][repr])]
        aggregated_metrics.append(aggregated)
    return aggregated_metrics


def sort_metrics_by_num_captions(aggregated_metrics, num_test_captions, samplings_per_amount):
    by_num_captions = {}
    for num_caps in range(num_test_captions):
        avg = {}
        for repr in aggregated_metrics[0].keys():
            avg[repr] = {}

            for metric in aggregated_metrics[num_caps * samplings_per_amount][repr].keys():
                for j in range(samplings_per_amount):
                    if metric not in avg[repr].keys():
                        avg[repr][metric] = []
                    avg[repr][metric].extend(aggregated_metrics[num_caps * samplings_per_amount + j][repr][metric])
                # avg[m][metric] = sum(avg[m][metric]) / len(avg[m][metric])

        by_num_captions[num_caps + 1] = avg
    return by_num_captions


def generate_and_save_plot(by_num_captions, save_dir, repr="sbert", metric="accuracy"):
    # Extract sbert values
    sbert_values = [by_num_captions[i][repr][metric] for i in by_num_captions.keys()]

    # Given mean and standard deviation lists
    mean_values = np.mean(sbert_values, axis=1)
    std_values = np.std(sbert_values, axis=1)

    # x-axis values
    x = np.arange(1, len(mean_values) + 1)

    # Plotting
    plt.errorbar(x, mean_values, yerr=std_values, fmt='o', capsize=4)
    plt.plot(x, mean_values, '-o')

    # Fill between line plot and error bars
    plt.fill_between(x, np.subtract(mean_values, std_values), np.add(mean_values, std_values), alpha=0.2)

    plt.savefig(f"{save_dir}/{repr}_{metric}.svg")
    plt.close()


def json_save(data, save_dir, filename):
    with open(f"{save_dir}/{filename}.json", "w") as f:
        json.dump(data, f)


def main(_):
    init_logging(logging.INFO)

    logging.info("Starting script")
    data_dir = FLAGS.data_dir
    load_dir = os.path.join(data_dir, FLAGS.load_dir)
    save_dir = os.path.join(data_dir, FLAGS.save_dir)

    dataset_info = FLAGS.dataset.split(",")
    assert len(dataset_info) == 2
    dataset_type = dataset_info[0]
    dataset_name = dataset_info[1]

    full_save_dir = os.path.join(save_dir, dataset_type, dataset_name)
    os.makedirs(full_save_dir, exist_ok=True)

    # set sampling parameters
    samplings_per_amount = FLAGS.samplings_per_amount
    num_test_captions = FLAGS.num_test_captions
    num_random_states = FLAGS.num_random_states if not FLAGS.debug else 2
    metrics = ["accuracy", "nmi", "ari"]
    random_states = random.sample(range(1000), num_random_states)
    json_save(random_states, full_save_dir, "random_states")
    # loading data

    logging.info("Loading model")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.encode(["First do it", "then do it right", "then do it better"])

    if FLAGS.debug == True:
        logging.info("Debug mode")
        dataset = load_from_disk(os.path.join(load_dir, dataset_type, dataset_name)).select(range(50))
    else:
        logging.info("Loading dataset")
        dataset = load_from_disk(os.path.join(load_dir, dataset_type, dataset_name))

    # start working on data
    id_list = list(range(len(dataset[0]["captions"])))
    sub_samples = generate_test_id_dict(
        id_list, num_test_captions=num_test_captions, samplings_per_amount=samplings_per_amount
    )
    json_save(sub_samples, full_save_dir, "sub_samples")

    logging.info("Getting embeddings and captions")
    full_np_embeddings, full_concatenated_captions = get_embeddings_and_captions(model, dataset, sub_samples)
    joblib.dump(full_np_embeddings, f"{full_save_dir}/full_np_embeddings.pbz2")
    joblib.dump(full_concatenated_captions, f"{full_save_dir}/full_concatenated_captions.pbz2")

    logging.info("Getting kmeans predictions")
    all_kmeans = embeddings_to_predictions(
        dataset, dataset_name, full_np_embeddings, full_concatenated_captions, random_states
    )
    joblib.dump(all_kmeans, os.path.join(full_save_dir, "all_kmeans.pbz2"))

    logging.info("Aggregating metrics")
    aggregated_metrics = aggregate_metrics(all_kmeans)
    json_save(aggregated_metrics, full_save_dir, "aggregated_metrics")
    by_num_captions = sort_metrics_by_num_captions(aggregated_metrics, num_test_captions, samplings_per_amount)
    json_save(by_num_captions, full_save_dir, "by_num_captions")

    logging.info("Generating plots")
    for metric in metrics:
        for repr in by_num_captions[1].keys():
            generate_and_save_plot(by_num_captions, full_save_dir, repr=repr, metric=metric)

    logging.info("Finished script")


if __name__ == "__main__":
    app.run(main)
