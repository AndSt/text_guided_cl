import os
import json
from datasets import load_from_disk


def fix_cifar10(dataset):
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    if "label" in dataset.features:
        dataset = dataset.remove_columns("label")

    def fix_label(example):
        example["label"] = label_names[example["cluster_label"]]
        return example

    dataset = dataset.map(fix_label)
    return dataset


def fix_human_action_recognition(dataset):
    label_names = [
        "calling", "clapping", "cycling", "dancing", "drinking", "eating", "fighting", "hugging", "laughing",
        "listening_to_music", "running", "sitting", "sleeping", "texting", "using_laptop"
    ]

    if "label" in dataset.features:
        dataset = dataset.remove_columns("label")

    def fix_label(example):
        example["label"] = label_names[example["cluster_label"]]
        return example

    dataset = dataset.map(fix_label)
    return dataset


def fix_sports10(dataset):
    label_names = [
        'AmericanFootball', 'Basketball', 'BikeRacing', 'CarRacing', 'Fighting',
        'Hockey', 'Soccer', 'TableTennis', 'Tennis', 'Volleyball'
    ]

    if "label" in dataset.features:
        dataset = dataset.remove_columns("label")

    def fix_label(example):
        example["label"] = label_names[example["cluster_label"]]
        return example

    dataset = dataset.map(fix_label)
    return dataset


def load_imagenet_synset():
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_synset_dict.json")
    with open(file) as f:
        json_dict = json.load(f)
    return json_dict


def fix_tiny_imagenet(dataset):
    tiny_imagenet_classes = [
        "n01443537", "n01629819", "n01641577", "n01644900", "n01698640", "n01742172", "n01768244", "n01770393",
        "n01774384", "n01774750", "n01784675", "n01882714", "n01910747", "n01917289", "n01944390", "n01950731",
        "n01983481", "n01984695", "n02002724", "n02056570", "n02058221", "n02074367", "n02094433", "n02099601",
        "n02099712", "n02106662", "n02113799", "n02123045", "n02123394", "n02124075", "n02125311", "n02129165",
        "n02132136", "n02165456", "n02226429", "n02231487", "n02233338", "n02236044", "n02268443", "n02279972",
        "n02281406", "n02321529", "n02364673", "n02395406", "n02403003", "n02410509", "n02415577", "n02423022",
        "n02437312", "n02480495", "n02481823", "n02486410", "n02504458", "n02509815", "n02666347", "n02669723",
        "n02699494", "n02769748", "n02788148", "n02791270", "n02793495", "n02795169", "n02802426", "n02808440",
        "n02814533", "n02814860", "n02815834", "n02823428", "n02837789", "n02841315", "n02843684", "n02883205",
        "n02892201", "n02909870", "n02917067", "n02927161", "n02948072", "n02950826", "n02963159", "n02977058",
        "n02988304", "n03014705", "n03026506", "n03042490", "n03085013", "n03089624", "n03100240", "n03126707",
        "n03160309", "n03179701", "n03201208", "n03255030", "n03355925", "n03373237", "n03388043", "n03393912",
        "n03400231", "n03404251", "n03424325", "n03444034", "n03447447", "n03544143", "n03584254", "n03599486",
        "n03617480", "n03637318", "n03649909", "n03662601", "n03670208", "n03706229", "n03733131", "n03763968",
        "n03770439", "n03796401", "n03814639", "n03837869", "n03838899", "n03854065", "n03891332", "n03902125",
        "n03930313", "n03937543", "n03970156", "n03977966", "n03980874", "n03983396", "n03992509", "n04008634",
        "n04023962", "n04070727", "n04074963", "n04099969", "n04118538", "n04133789", "n04146614", "n04149813",
        "n04179913", "n04251144", "n04254777", "n04259630", "n04265275", "n04275548", "n04285008", "n04311004",
        "n04328186", "n04356056", "n04366367", "n04371430", "n04376876", "n04398044", "n04399382", "n04417672",
        "n04456115", "n04465666", "n04486054", "n04487081", "n04501370", "n04507155", "n04532106", "n04532670",
        "n04540053", "n04560804", "n04562935", "n04596742", "n04598010", "n06596364", "n07056680", "n07583066",
        "n07614500", "n07615774", "n07646821", "n07647870", "n07657664", "n07695742", "n07711569", "n07715103",
        "n07720875", "n07749582", "n07753592", "n07768694", "n07871810", "n07873807", "n07875152", "n07920052",
        "n07975909", "n08496334", "n08620881", "n08742578", "n09193705", "n09246464", "n09256479", "n09332890",
        "n09428293", "n12267677", "n12520864", "n13001041", "n13652335", "n13652994", "n13719102", "n14991210"
    ]

    json_dict = load_imagenet_synset()

    def fix_label(example):
        imgnet_class = tiny_imagenet_classes[example["cluster_label"]]
        words = json_dict[imgnet_class]["words"]
        example["label"] = words
        return example

    dataset = dataset.map(fix_label)
    return dataset


def fix_food101(dataset):
    food101_classes = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets",
        "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad",
        "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings",
        "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee",
        "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
        "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast",
        "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
        "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog",
        "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
        "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters",
        "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine",
        "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
        "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
        "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
        "waffles"]

    def fix_label(example):
        food101_class = food101_classes[example["cluster_label"]]
        example["label"] = food101_class
        return example

    dataset = dataset.map(fix_label)
    return dataset


def fix_imagenet10(dataset):
    json_dict = load_imagenet_synset()

    if "label" in dataset.features:
        dataset = dataset.remove_columns("label")

    def fix_label(example):
        imgnet_class = example["cluster_label"]
        words = json_dict[imgnet_class]["words"]
        example["label"] = words
        return example

    dataset = dataset.map(fix_label)
    return dataset


def add_label_and_save(data_dir, dataset_name):
    dataset = load_from_disk(os.path.join(data_dir, dataset_name))

    if dataset_name in ["stl10", "lsun", "inat", "fer2013"]:
        print(f"Skipping {dataset_name} ause it's already good")
    elif dataset_name == "cifar10":
        print("Fixing cifar10")
        dataset = fix_cifar10(dataset)
    # elif dataset_name == "tiny_imagenet":
    #     dataset = fix_tiny_imagenet(dataset)
    elif dataset_name == "imagenet10":
        print("Fixing imagenet10")
        dataset = fix_imagenet10(dataset)
    elif dataset_name == "human_action_recognition":
        print("Fixing human_action_recognition")
        dataset = fix_human_action_recognition(dataset)
    elif dataset_name == "sports10":
        print("Fixing sports10")
        dataset = fix_sports10(dataset)
    else:
        print(f"Skipping {dataset_name} because it's not supported")

    return dataset
