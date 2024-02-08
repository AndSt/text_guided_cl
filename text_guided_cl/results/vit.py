import os
import joblib

import numpy as np


def load_dset_vit(vit_dir, dataset_type, dataset_name, metrics=["accuracy", "nmi"], num_subsets=5):
    vit_file = os.path.join(vit_dir, dataset_type, dataset_name, f"all_kmeans.pbz2")
    vit_data = joblib.load(vit_file)

    if num_subsets is None:
        num_subsets = len(vit_data)

    dataset_summary = {}
    for metric in metrics:
        metric_list = []
        for j in range(0, len(vit_data), num_subsets):

            value_list = []
            inertia_list = []
            for i, clustering in enumerate(vit_data[j * num_subsets: (j + 1) * num_subsets]):
                if len(list(clustering.keys())) == 0:
                    print(f"Empty clustering {i}")
                    continue
                value_list.append(clustering["metrics"][metric])
                inertia_list.append(clustering["inertia"])

            if len(value_list) == 0:
                continue
            idx = np.argmin(inertia_list)
            metric_list.append(value_list[idx])
        dataset_summary[metric] = sum(metric_list) / len(metric_list)

    return dataset_summary


# def load_full_vit(vit_dir, metrics=["accuracy", "nmi"], type="inertia"):
#     full_vit = {}
#     for dataset_type in os.listdir(vit_dir):
#         dataset_type_desc = dataset_type[0:-len("_caption")]
#         full_vit[dataset_type_desc] = {}
#         for dataset_name in os.listdir(os.path.join(vit_dir, dataset_type)):
#             vit_file = os.path.join(vit_dir, dataset_type, dataset_name, f"all_kmeans.pbz2")
#             vit_data = joblib.load(vit_file)
#             dataset_summary = {}
#             for metric in metrics:
#                 value_list = []
#                 inertia_list = []
#                 for i, clustering in enumerate(vit_data):
#                     if len(list(clustering.keys())) == 0:
#                         print(f"Empty clustering {i}")
#                         continue
#                     value_list.append(clustering["metrics"][metric])
#                     inertia_list.append(clustering["inertia"])
#                 if type == "inertia":
#                     idx = np.argmin(inertia_list)
#                     dataset_summary[metric] = value_list[idx]
#                 elif type == "mean":
#                     dataset_summary[metric] = sum(value_list) / len(value_list)
#                 elif type == "median":
#                     dataset_summary[metric] = np.median(value_list)
#                 elif type == "max":
#                     dataset_summary[metric] = max(value_list)
#                 elif type == "min":
#                     dataset_summary[metric] = min(value_list)
#                 else:
#                     raise ValueError(f"Unknown type {type}")
#             full_vit[dataset_type_desc][dataset_name] = dataset_summary
#     return full_vit


def load_all_inertia_vit(vit_dir, metrics=["accuracy", "nmi"], num_subsets=5):
    full_vit = {}
    for dataset_type in os.listdir(vit_dir):
        if dataset_type.endswith("_caption"):
            dataset_type_desc = dataset_type[:-len("_caption")]
        elif dataset_type.endswith("_vqa"):
            dataset_type_desc = dataset_type.split("_vqa")[0]
        full_vit[dataset_type_desc] = {}
        for dataset_name in os.listdir(os.path.join(vit_dir, dataset_type)):
            dataset_summary = load_dset_vit(
                vit_dir, dataset_type, dataset_name, metrics=metrics, num_subsets=num_subsets
            )
            full_vit[dataset_type_desc][dataset_name] = dataset_summary
    return full_vit
