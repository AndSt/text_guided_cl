import os
import json

from slurm_utils.convenience.flags import flags, FLAGS

from absl import app

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


flags.DEFINE_string("data_dir", default="")
flags.DEFINE_string("model_name", default="all-MiniLM-L6-v2", help="Model name. It needs to be a CausalLM")
flags.DEFINE_string("load_dir", default="blip2_captions", help="Number of keyword generations per cluster")
flags.DEFINE_string("save_dir", default="sbert_embeddings", help="Number of keyword generations per cluster")
flags.DEFINE_string("dataset", default="stl_10", help="Number of keyword generations per cluster")
flags.DEFINE_bool("unconcatenated", default=False, help="Whether the captions are also embedded unconcatenated.")


def main(_):
    model = SentenceTransformer(FLAGS.model_name)

    def add_emb(sample):
        sample["sbert_embedding"] = model.encode(". ".join(sample["captions"]))
        if FLAGS.unconcatenated:
            for i in range(len(sample["captions"])):
                sample[f"sbert_embedding_{i}"] = model.encode(sample["captions"][i])
        return sample

    dataset = load_from_disk(os.path.join(FLAGS.data_dir, FLAGS.load_dir, FLAGS.dataset))

    dataset = dataset.map(add_emb)

    save_dir = os.path.join(FLAGS.data_dir, FLAGS.save_dir, FLAGS.dataset)
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)


if __name__ == '__main__':
    app.run(main)
