import os
import joblib

from slurm_utils.convenience.flags import flags, FLAGS

from absl import app

from text_guided_cl.evaluate.embeddings import  create_all_kmeans_predictions

flags.DEFINE_string("data_dir", default="blip2_sbert", help="Number of keyword generations per cluster")
flags.DEFINE_string("load_dir", default="blip2_sbert", help="Number of keyword generations per cluster")
flags.DEFINE_string("save_dir", default="blip2_sbert_eval", help="Number of keyword generations per cluster")



def main(_):

    data_dir = os.path.join(FLAGS.data_dir, FLAGS.load_dir)
    load_dirs = os.listdir(data_dir)

    random_states = [24, 48, 72, 96, 120]

    for load_dir in load_dirs:
        for dataset in os.listdir(os.path.join(data_dir, load_dir)):

            dataset_predictions = create_all_kmeans_predictions(
                load_dir=os.path.join(data_dir, load_dir),
                dataset=dataset,
                random_states=random_states
            )

            save_dir = os.path.join(FLAGS.data_dir, FLAGS.save_dir, load_dir, dataset)
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(dataset_predictions, os.path.join(save_dir, "predictions.pbz2"))



if __name__ == '__main__':
    app.run(main)
