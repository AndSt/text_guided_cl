import os
import json
from typing import Union, List

import logging
from absl import app, flags

from PIL import Image
from datasets import load_from_disk

import torch
from einops import repeat
from flamingo_mini import FlamingoProcessor, FlamingoModel

from text_guided_cl.logging import init_logging

from slurm_utils.convenience.flags import flags, FLAGS

flags.DEFINE_string("dataset", default="fashion_mnist", help="Model name. It needs to be a CausalLM")
flags.DEFINE_string("load_dir", default="", help="Directory to load the model from.")
flags.DEFINE_string("save_dir", default="", help="Directory to save the model to.")
flags.DEFINE_string("model_name", default="dhansmair/flamingo-mini", help="Model name. It needs to be a CausalLM")
flags.DEFINE_integer("captions_per_sample", default=5, help="Number of keyword generations per cluster")
flags.DEFINE_integer("max_length", default=20, help="Number of samples used to describe the cluster")


@torch.no_grad()
def generate_captions(
        model: FlamingoModel,
        processor: FlamingoProcessor,
        pixel_values: Union[torch.Tensor, None] = None,
        sample = None,
        prompt: str = "<image>",
        max_length: int = 150,
        do_sample: bool = True,
        top_k: int = 4,
        num_return_sequences: int = 5,
        device: Union[torch.device, None] = None
):
    """
    helper utility for image captioning.
    prompt is replicated for all batches.
    """
    images = sample["image"]
    if images is not None:
        assert pixel_values is None, "you can only pass either images or visual features to generate_captions()!"

        if isinstance(images, Image.Image):
            images = [images]

        pixel_values = processor(images=images, device=device)['pixel_values']

    assert pixel_values is not None, "you must pass either images or visual features to generate_captions()!"

    batch_size = pixel_values.size(0)
    assert batch_size == 1, "only one image can be passed at a time"
    input_ids, media_locations, attention_mask = processor.encode_text(prompt, device)
    input_ids = repeat(input_ids[0], 'l -> n l', n=batch_size)
    media_locations = repeat(media_locations[0], 'l -> n l', n=batch_size)
    attention_mask = repeat(attention_mask[0], 'l -> n l', n=batch_size)

    out_ids = model.generate(
        inputs=input_ids,
        media_locations=media_locations,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        # num_beams=1,
        early_stopping=True,
        use_cache=True,
        bos_token_id=model.flamingo.lm.config.bos_token_id,
        eos_token_id=model.flamingo.lm.config.eos_token_id,
        pad_token_id=model.flamingo.lm.config.eos_token_id,
        max_length=max_length,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
    ).view(num_return_sequences, -1)

    caption_texts = [processor.remove_tags(processor.tokenizer.decode(ids, skip_special_tokens=True)) for ids in
                     out_ids]

    output = model.flamingo.vision_encoder(pixel_values).pooler_output

    sample["captions"] = caption_texts
    sample["visual_embedding"] = output[0]
    return sample


def main(_):
    init_logging()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Data loaded.")

    model = FlamingoModel.from_pretrained(FLAGS.model_name)
    model.to(device)
    model.eval()
    processor = FlamingoProcessor(model.config)
    logging.info("Model loaded.")

    def generate_descriptions(sample, num_captions: int = 5, max_length: int = 30):
        sample = generate_captions(
            model=model,
            processor=processor,
            sample=sample,
            num_return_sequences=num_captions,
            max_length=max_length,
            device=device
        )
        return sample

    logging.info("Start captioning.")
    dataset = load_from_disk(os.path.join(FLAGS.data_dir, FLAGS.load_dir, FLAGS.dataset))
    dataset = dataset.map(generate_descriptions, fn_kwargs={
        "num_captions": FLAGS.captions_per_sample,
        "max_length": FLAGS.max_length
    }, )
    logging.info("Captioning finished.")

    save_dir = os.path.join(FLAGS.data_dir, FLAGS.save_dir, FLAGS.dataset)
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)


if __name__ == '__main__':
    app.run(main)
