## Text-Guided Image Clustering


Authors: Andreas Stephan, Vasiliki Kougia, Benjamin Roth

This repo contains the code related to the paper "Text-Guided Image Clustering", accepted to the main conference at EACL 2024.

If there's questions, please contact us at [andreas.stephan@univie.ac.at](mailto:andreas.stephan@univie.ac.at)

---------

## Abstract

Image clustering divides a collection of images into meaningful groups, typically interpreted post-hoc via human-given annotations. Those are usually in the form of text, begging the question of using text as an abstraction for image clustering. Current image clustering methods, however, neglect the use of generated textual descriptions. We, therefore, propose \textit{Text-Guided Image Clustering}, i.e. generating text using image captioning and visual question-answering (VQA) models, and subsequently clustering the generated text. Further, we introduce a novel approach to inject task- or domain knowledge for clustering by prompting VQA models. Across eight diverse image clustering datasets, our results show that the obtained text representations often outperform image features. Additionally, we propose a counting-based cluster explainability method. Our evaluations show that the derived keyword-based explanations describe clusters better than the respective cluster accuracy suggests. Overall, this research challenges traditional approaches and paves the way for a paradigm shift in image clustering, using generated text.

-----

## Installation

There are two requirement files. If you want to explicitly work with our versions, run ```pip install fixed_requirements.txt```
In case you want to work with newer version of the used libraries, run ```pip install requirements.txt```. Note that we do not continuously test whether this works.

Finally you should also run ```pip install -e .``` to make sure imports work.

### Download data

Find the data, including generated text here:

## Usage

If anything is unclear, don't hesitate, contact us immediately [andreas.stephan@univie.ac.at](mailto:andreas.stephan@univie.ac.at).

### Image-to-text

In order to run captioning, there is an example config in ```caption/blip2_vqa/config.cfg```, so you can run it via

```bash
PYTHONPATH=. python caption/blip2.py  --flagfile=examples/blip2_vqa/config.cfg
```

Make sure to change the data folder paths first.

### SentenceBert

To add SentenceBert embeddings to a dataset, you just need to set the "data_dir" + "load_dir" arguments to the previous "save_dir"

```bash
PYTHONPATH=. python embed_sentence_bert.py  --flagfile=examples/blip2_vqa/config.cfg
```

### KMeans Evaluation

```bash
PYTHONPATH=. python embed_sentence_bert.py  --flagfile=examples/blip2_vqa/config.cfg
```

## Citation

```
@misc{stephan2024textguided,
      title={Text-Guided Image Clustering}, 
      author={Andreas Stephan and Lukas Miklautz and Kevin Sidak and Jan Philip Wahle and Bela Gipp and Claudia Plant and Benjamin Roth},
      year={2024},
      eprint={2402.02996},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```