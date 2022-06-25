# Initial setup to be able to load `src.cyclegan`
import sys
import os
from numpy import newaxis
import logging
from argparse import ArgumentParser

from utils import load_data, save_midis
from cyclegan import CycleGAN

logger = logging.getLogger("convert_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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
    args.add_argument("model_path", type=str, help="Path to a trained CycleGAN model.")
    args.add_argument(
        "--direction",
        type=str,
        default="A2B",
        help="Transfer direction.",
        choices=["A2B", "B2A"],
    )
    args.add_argument(
        "--set_type",
        type=str,
        default="test",
        help="Name of folder container files to convert.",
    )
    args.add_argument(
        "--outpath", default="converted", type=str, help="Path to output location."
    )

    return args.parse_args(argv)


def main(argv):
    """The main function for training."""
    args = parse_args(argv)
    path_a = args.path_a
    path_b = args.path_b
    model_path = args.model_path
    model_name = model_path.split("/")[-1]
    model_fpath = os.path.join(os.getcwd(), model_path, "weights", "")

    direction = args.direction
    outpath = args.outpath
    set_type = args.set_type

    # debug args
    path_a = "data/JC_C_cp/tfrecord"  # dummy dir with less data
    path_b = "data/JC_J_cp/tfrecord"
    model_path = "trained_models"
    model_name = "classic2jazz_15e_bs32_run_2022_06_22-20_08_16"
    model_fpath = os.path.join(os.getcwd(), model_path, model_name, "weights", "")

    # TODO: Handle this a bit more cleanly
    genre_a = path_a.split("/")[1]
    genre_b = path_b.split("/")[1]
    if genre_a == path_a or genre_b == path_b:
        raise Exception(
            "There was an issue parsing the genre from the filename. Check the separator used."
        )
    dataset = load_data(path_a, path_b, set_type, shuffle=False)

    # Setup model
    logger.debug(f"Loading model from {model_fpath}")
    model = CycleGAN(genre_a, genre_b)
    model.load_weights(model_fpath)
    logger.debug(f"\tsuccess!")

    outpath = f"converted/{model_name}"
    os.makedirs(outpath, exist_ok=True)
    logger.debug(f"Converting and saving results to {outpath}")
    for batch in dataset:
        original_inputs, converted, cycled = model(batch, direction=direction)

        for idx, (original, transfer, cycle) in enumerate(
            zip(original_inputs, converted, cycled)
        ):
            save_midis(original[newaxis, ...], f"{outpath}/{idx}_original.mid")
            save_midis(transfer[newaxis, ...], f"{outpath}/{idx}_transfer.mid")
            save_midis(cycle[newaxis, ...], f"{outpath}/{idx}_cycle.mid")


if __name__ == "__main__":
    main(sys.argv[1:])
