import os
import json

import logging
from absl import app

from datasets import load_from_disk

import torch
from transformers import Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM

from text_guided_cl.logging import init_logging
from slurm_utils.convenience.flags import flags, FLAGS

flags.DEFINE_string("dataset", default="fashion_mnist", help="Model name. It needs to be a CausalLM")
flags.DEFINE_string("load_dir", default="", help="Directory to load the model from.")
flags.DEFINE_string("save_dir", default="", help="Directory to save the model to.")
flags.DEFINE_string("model_name", default="microsoft/git-large-coco",
                    help="Model name. It needs to be a CausalLM")
flags.DEFINE_integer("captions_per_sample", default=5, help="Number of keyword generations per cluster")
flags.DEFINE_integer("max_length", default=20, help="Number of samples used to describe the cluster")


@torch.no_grad()
def generate_captions(processor, model, sample, device, text: str = None, max_length=200, num_return_sequences=5):
    if text is not None:
        inputs = processor(images=sample["image"], text=text, return_tensors="pt").to(device)  # , torch.float16)
    else:
        inputs = processor(images=sample["image"], return_tensors="pt").to(device)  # , torch.float16)

    generated_ids = model.generate(
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        **inputs
        # top_k=8,
    )

    sample["captions"] = processor.batch_decode(generated_ids, skip_special_tokens=True)
    sample["visual_embedding"] = model.git.image_encoder(
        pixel_values=inputs.pixel_values
    ).last_hidden_state.cpu().sum(dim=1).numpy().tolist()
    return sample


def main(_):
    init_logging()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Data loaded.")

    git_processor_large = AutoProcessor.from_pretrained(FLAGS.model_name)
    git_model_large = AutoModelForCausalLM.from_pretrained(FLAGS.model_name)

    git_model_large.to(device)
    logging.info("Model loaded.")

    def generate_descriptions(sample, num_captions: int = 5, max_length: int = 30):
        sample2 = generate_captions(
            model=git_model_large,
            processor=git_processor_large,
            sample=sample,
            num_return_sequences=num_captions,
            max_length=max_length,
            device=device
        )
        return sample2

    logging.info("Start captioning.")
    dataset = load_from_disk(os.path.join(FLAGS.data_dir, FLAGS.load_dir, FLAGS.dataset))
    dataset = dataset.map(generate_descriptions, fn_kwargs={
        "num_captions": FLAGS.captions_per_sample,
        "max_length": FLAGS.max_length
    }, load_from_cache_file=False)
    logging.info("Captioning finished.")

    save_dir = os.path.join(FLAGS.data_dir, FLAGS.save_dir, FLAGS.dataset)
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)

    # with open(os.path.join(FLAGS.work_dir, "test_metrics.json"), "w") as f:
    #     json.dump({"accuracy": 0.5}, f)


if __name__ == '__main__':
    app.run(main)
