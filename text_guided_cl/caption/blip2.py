import os
import json

import logging
from absl import app

from datasets import load_from_disk

import torch
from transformers import Blip2ForConditionalGeneration, AutoProcessor

from text_guided_cl.logging import init_logging
from slurm_utils.convenience.flags import flags, FLAGS

flags.DEFINE_string("dataset", default="fashion_mnist", help="Model name. It needs to be a CausalLM")
flags.DEFINE_string("load_dir", default="", help="Directory to load the model from.")
flags.DEFINE_string("save_dir", default="", help="Directory to save the model to.")
flags.DEFINE_string("model_name", default="Salesforce/blip-image-captioning-large",
                    help="Model name. It needs to be a CausalLM")
flags.DEFINE_integer("captions_per_sample", default=5, help="Number of keyword generations per cluster")
flags.DEFINE_integer("max_length", default=40, help="Number of samples used to describe the cluster")
flags.DEFINE_list("question", default=None, help="Question to ask the model about the image.")


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
    sample["visual_embedding"] = model.vision_model(inputs.pixel_values, return_dict=True).pooler_output[
        0].cpu().numpy().tolist()
    return sample


def assert_question_list_flag():
    if FLAGS.question is not None and FLAGS.question is not "":
        assert isinstance(FLAGS.question, list), "Question flag must be a list of strings"
        assert len(FLAGS.question) == 2, "Question flag must be a list of strings of length 1"
        assert FLAGS.question[0] != ""
        assert FLAGS.question[1] != ""
        return FLAGS.question[0], FLAGS.question[1]
    return None, None


def main(_):
    init_logging()

    question_dir, question = assert_question_list_flag()
    logging.info(f"Question: {question_dir} {question}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Data loaded.")

    blip2_processor_large = AutoProcessor.from_pretrained(FLAGS.model_name)
    blip2_model_large = Blip2ForConditionalGeneration.from_pretrained(FLAGS.model_name)  # , torch_dtype=torch.float16)
    blip2_model_large.to(device)
    logging.info("Model loaded.")

    def generate_descriptions(sample, num_captions: int = 5, max_length: int = 30):
        sample2 = generate_captions(
            model=blip2_model_large,
            processor=blip2_processor_large,
            sample=sample,
            num_return_sequences=num_captions,
            max_length=max_length,
            device=device,
            text=question
        )
        return sample2

    logging.info("Start captioning.")
    dataset = load_from_disk(os.path.join(FLAGS.data_dir, FLAGS.load_dir, FLAGS.dataset))
    dataset = dataset.map(generate_descriptions, fn_kwargs={
        "num_captions": FLAGS.captions_per_sample,
        "max_length": FLAGS.max_length
    }, load_from_cache_file=False)
    logging.info("Captioning finished.")

    if question_dir is not None:
        save_dir = os.path.join(FLAGS.data_dir, f"{FLAGS.save_dir}_vqa_{question_dir}", FLAGS.dataset)
    else:
        save_dir = os.path.join(FLAGS.data_dir, f"{FLAGS.save_dir}_caption", FLAGS.dataset)

    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)


if __name__ == '__main__':
    app.run(main)
