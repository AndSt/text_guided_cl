import numpy as np


def string_match(pred_df, exact_match=True):

    def check_match(row):
        pred_labels = row["pred"].split(",")
        pred_labels = [x.strip().lower() for x in pred_labels]

        truth_label = row["truth"].lower()
        truth_label = truth_label.replace("_", " ")

        for j in range(len(pred_labels)):
            if exact_match:
                if pred_labels[j] == truth_label:
                    return 1
            else:
                if pred_labels[j] in truth_label or truth_label in pred_labels[j]:
                    return 1
        return 0

    pred_df["match"] = pred_df.apply(check_match, axis=1)
    return pred_df
