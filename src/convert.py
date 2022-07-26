# Initial setup to be able to load `src.cyclegan`
import sys
import os
from numpy import newaxis
import logging
from argparse import ArgumentParser
from yaml import safe_load, YAMLError

from utils import load_data, save_midis
from cyclegan import CycleGAN


logging.basicConfig(
    filename="converter.log",
    format="%(asctime)s : %(name)s [%(levelname)s] : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("convert_logger")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Suppress tensorflow logs


def parse_args(argv):
    """Parses input options for this module.

    Parameters
    ----------
    argv : List
        List of input parameters to this module

    Returns
    -------
    Argparser
    """
    args = ArgumentParser()
    args.add_argument("path_a", type=str, help="Path to tfrecord files of dataset A.")
    args.add_argument("path_b", type=str, help="Path to tfrecord files of dataset B.")
    args.add_argument("genre_a", type=str, help="Name of genre A.")
    args.add_argument("genre_b", type=str, help="Name of genre B.")
    args.add_argument("model_path", type=str, help="Path to a trained CycleGAN model.")
    args.add_argument(
        "--set_type",
        type=str,
        default="test",
        help="Name of folder container files to convert.",
    )
    args.add_argument(
        "--config_fpath",
        default="src/config.yaml",
        type=str,
        help="Path to YAML file containing model configuration.",
    )
    args.add_argument(
        "--outpath", default="converted", type=str, help="Path to output location."
    )

    return args.parse_args(argv)


def convert_batch(model, batch, outpath, direction, idx):
    original_inputs, converted, cycled = model(batch, direction=direction)
    for (original, transfer, cycle) in zip(original_inputs, converted, cycled):
        base_name = f"{outpath}/{direction}/{idx}"

        save_midis(original, f"{base_name}_original.mid")
        save_midis(transfer, f"{base_name}_transfer.mid")
        save_midis(cycle, f"{base_name}_cycle.mid")

        idx += 1
    return idx


def load_config(config_path):
    """Loads the model config from the given path

    Parameters
    ----------
    config_path : str
        Path to yaml file containing model configuration parameters.
    """
    with open(config_path, "r") as config_file:
        try:
            config = safe_load(config_file)
        except YAMLError as e:
            raise e
    return config["CycleGAN"]


def main(argv):
    """The main function for performing style transfer."""
    args = parse_args(argv)
    path_a = args.path_a
    path_b = args.path_b
    genre_a = args.genre_a
    genre_b = args.genre_b
    model_path = args.model_path
    set_type = args.set_type
    config_fpath = args.config_fpath
    outpath = args.outpath

    model_name = model_path.split("/")[-1]
    model_fpath = os.path.join(os.getcwd(), model_path, "weights", "")

    logger.info(
        "#" * 20 + f" Converting {genre_a}2{genre_b} with: {model_name} " + "#" * 20
    )

    dataset = load_data(path_a, path_b, set_type, shuffle=False)

    model_config = load_config(config_fpath)
    n_timesteps = model_config["n_timesteps"]

    # Setup model
    logger.info(f"Loading model from {model_fpath}")
    model = CycleGAN(genre_a, genre_b, **model_config)
    model.load_weights(model_fpath)
    logger.info(f"\tsuccess!")

    outpath = f"{outpath}/{model_name}"
    os.makedirs(f"{outpath}/A2B", exist_ok=True)
    os.makedirs(f"{outpath}/B2A", exist_ok=True)
    logger.info(f"Converting and saving results to {outpath}")

    idx_a2b, idx_b2a = 0, 0
    for batch in dataset:
        idx_a2b = convert_batch(model, batch, outpath, "A2B", idx_a2b)
        idx_b2a = convert_batch(model, batch, outpath, "B2A", idx_b2a)
    logger.info(f"[Done]")


if __name__ == "__main__":
    main(sys.argv[1:])
