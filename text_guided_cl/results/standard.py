import os
import json
import joblib

import numpy as np
import pandas as pd


def get_run_dir_to_name_dict():
    run_dir_to_name_dict = {
        "no_captions": "Human",
        "no_captions_standard": "Standard",
        "no_captions_hq": "HQ",
        "no_captions_big": "Large No.",
    }
    return run_dir_to_name_dict


def get_dataset_id2name_dict():
    dataset_id2name = {
        "fer2013": "FER2013",
        "lsun": "LSUN",
        "human_action_recognition": "HAR",
        "inat": "iNaturalist2021",
        "sports10": "Sports10",
        "places365_val_subset": "Places365",
        "food101": "Food101",
        "cifar10": "Cifar10",
        "stl10": "STL10",
        "imagenet10": "ImageNet10",
        "tiny_imagenet": "TinyImageNet",
        "avg": "Avg"
    }
    return dataset_id2name


def extract_from_num_caption_analysis(
        caption_analysis_dir, dataset_type, dataset_name, num_captions=4, num_versions=1,
        metrics=["accuracy", "nmi", "ari"], aggr_func="mean"
):
    dataset_summary = {}

    num_captions_file = os.path.join(caption_analysis_dir, dataset_type, dataset_name, "by_num_captions.json")
    with open(num_captions_file, "r") as f:
        num_captions_data = json.load(f)[f"{num_captions}"]

    for repr in list(num_captions_data.keys()):
        dataset_summary[repr] = {}
        for metric in metrics:
            value_list = num_captions_data[repr][metric][0:num_versions]
            if aggr_func == "mean":
                dataset_summary[repr][metric] = sum(value_list) / len(value_list)
            elif aggr_func == "max":
                dataset_summary[repr][metric] = max(value_list)
            elif aggr_func == "min":
                dataset_summary[repr][metric] = min(value_list)
            else:
                raise ValueError(f"Unknown aggregation function {aggr_func}")

    return dataset_summary


def extract_inertia_from_all_kmeans(
        analyze_dir, dataset_type, dataset_name, num_captions=4, num_versions=1,
        metrics=["accuracy", "nmi", "ari"], aggr_func="mean", num_subsets=None
):
    load_dir = os.path.join(analyze_dir, dataset_type, dataset_name)
    all_kmeans = joblib.load(os.path.join(load_dir, "all_kmeans.pbz2"))

    if num_subsets is None:
        num_subsets = len(all_kmeans[0])

    # open sub_samples.json
    with open(os.path.join(load_dir, "sub_samples.json"), "r") as f:
        sub_samples = json.load(f)

    samples_per_num_captions = len(sub_samples["1"])

    dataset_summary = {}
    for repr in ["tfidf", "sbert"]:
        dataset_summary[repr] = {}
        for i in range(0, max(num_versions, samples_per_num_captions)):

            value_list = {metric: [] for metric in metrics}

            idx = (num_captions - 1) * samples_per_num_captions + i

            for j in range(0, len(all_kmeans[idx]), num_subsets):
                # print(j, len(all_kmeans[idx]), "we are here")
                best_run = -1
                best_random_state = -1
                best_inertia = 0

                for run_id, run in enumerate(all_kmeans[idx][j * num_subsets: (j + 1) * num_subsets]):
                    # print(f"This is run {run_id}")
                    run = run[repr]
                    if run["inertia"] < best_inertia or best_random_state == -1:
                        best_random_state = run["random_state"]
                        best_inertia = run["inertia"]
                        best_run = run_id

                prediction_dict = all_kmeans[idx][best_run][repr]

                for metric in metrics:
                    value_list[metric].append(prediction_dict["metrics"][metric])

            for metric in metrics:
                if  metric not in dataset_summary[repr]:
                    dataset_summary[repr][metric] = []
                dataset_summary[repr][metric].append(sum(value_list[metric]) / len(value_list[metric]))

        for metric in metrics:
            if aggr_func == "mean":
                dataset_summary[repr][metric] = sum(dataset_summary[repr][metric]) / len(dataset_summary[repr][metric])
            elif aggr_func == "max":
                dataset_summary[repr][metric] = max(dataset_summary[repr][metric])
            elif aggr_func == "min":
                dataset_summary[repr][metric] = min(dataset_summary[repr][metric])

    return dataset_summary


def load_full_info(
        analysis_dir, num_captions=4, num_versions=1, metrics=["accuracy", "nmi", "ari"], aggr_func="mean",
        num_subsets=5
):
    kw_datasets = {}

    for dataset_type in os.listdir(analysis_dir):
        if dataset_type not in kw_datasets:
            kw_datasets[dataset_type] = {}
        for dataset_name in os.listdir(os.path.join(analysis_dir, dataset_type)):
            if dataset_name not in kw_datasets[dataset_type]:
                kw_datasets[dataset_type][dataset_name] = {}

            if len(os.listdir(os.path.join(analysis_dir, dataset_type, dataset_name))) != 13:
                print(f"Empty dataset {dataset_type} {dataset_name}")
                continue
            print(f"Loading dataset {dataset_type} {dataset_name}")
            dataset_summary = extract_inertia_from_all_kmeans(
                analysis_dir, dataset_type, dataset_name,
                num_captions=num_captions, num_versions=num_versions, metrics=metrics, aggr_func=aggr_func,
                num_subsets=num_subsets
            )
            kw_datasets[dataset_type][dataset_name] = dataset_summary
    return kw_datasets
