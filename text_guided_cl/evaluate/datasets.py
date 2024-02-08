def get_num_classes_dict():
    num_classes = {
        "fashion_mnist": 10,
        "cifar10": 10,
        "cifar100": 20,
        "tiny_imagenet": 200,
        "stl10": 10,
        "imagenet10": 10,
        "imagenet1k": 1000,
        "metashift_persons": 10,
        "imagenet1k_val": 1000,
        "places365_val_subset": 365,
        "inaturalist_val_subset": 15,
        "inat": 11,
        "human_action_recognition": 15,
        "dtd": 47,
        "fer2013": 7,
        "lsun": 10,
        "food101": 101,
        "sports10": 10,
    }
    return num_classes


run_dir_to_name_dict = {
    "no_captions_standard": "Standard",
    "no_captions_big": "Large No.",
    "no_captions_hq": "Background",
    "no_captions": "Human",
}

dataset_id2name = {
    "fer2013": "FER2013",
    "lsun": "LSUN",
    "human_action_recognition": "HAR",
    "inat": "iNaturalist2021",
    "sports10": "Sports10",
    "food101": "Food101",
    "cifar10": "CIFAR10",
    "stl10": "STL10",
    "imagenet10": "ImageNet10",
}