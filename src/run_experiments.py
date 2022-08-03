"""
    Trains a number of CycleGAN models for experimentation.

    A few things are hardcored in this script as I didn't have much time to tidy it up.
    The experiments are evaluated against the bodhidharma dataset (assumed to be in ../data/bodhidharma).
    
    Usage
    -----
    To run experiments for the original CycleGAN datasets in the `data/CycleGAN` folder
    use:
        python src/run_experiments.py \\
            data/CycleGAN \\
            genre_a \\
            genre_b \\
            bodhidharma_genre_a \\
            bodhidharma_genre_b
    
    This assumes that the `data/CycleGAN` directory contains folders of the form
        `tfrecord/n_bars/genre_a`
        `tfrecord/n_bars/genre_b`
    with the prepared data.
"""
import sys
import os
import logging
from datetime import datetime
from itertools import product
from yaml import safe_load, YAMLError, dump
from argparse import ArgumentParser

from prepare_data import main as prep_data
from train import main as train_pipeline
from convert import main as style_transfer
from evaluate import main as eval_experiment

logging.basicConfig(
    filename="experiments.log",
    format="%(asctime)s : %(name)s [%(levelname)s] : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("experiment_logger")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress tensorflow warning logs


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
    args.add_argument("root_path", type=str, help="Path to where the midis are stored.")
    args.add_argument("genre_a", type=str, help="Name of genre A.")
    args.add_argument("genre_b", type=str, help="Name of genre B.")
    args.add_argument("bd_genre_a", type=str, help="Name of Bodhidharma genre A.")
    args.add_argument("bd_genre_b", type=str, help="Name of Bodhidharma genre B.")
    return args.parse_args(argv)


def load_config(config_path):
    """Loads the model config from the given path

    Parameters
    ----------
    config_path : str
        Path to yaml file containing model configuration parameters.

    Returns
    -------
    dict
    """
    with open(config_path, "r") as config_file:
        try:
            config = safe_load(config_file)
        except YAMLError as e:
            raise e
    return config


def dict_cartesian_product(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))


def get_latest_trained_model():
    logger.info(f"Finding most recently trained model")
    model_info = [
        (fpath, datetime.strptime(fpath[-19:], "%Y_%m_%d-%H_%M_%S"))
        for fpath in os.listdir("trained_models")
    ]
    recent = sorted(model_info, key=lambda info: info[1])[-1]
    logger.info(f"Found {recent[0]}")
    return recent[0]


def main(argv):
    args = parse_args(argv)
    root_path = args.root_path
    genre_a = args.genre_a
    genre_b = args.genre_b

    bd_genre_a = args.bd_genre_a
    bd_genre_b = args.bd_genre_b

    config_path = "src/config.yaml"
    n_bars = [1, 2, 4, 8]
    batch_sizes = [64, 48, 32, 16]
    config_outfile = "src/train_config.yaml"

    bodhidharma_path = "../data/bodhidharma"
    convert_outpath = "converted/bodhidharma"

    results_outpath = "results/bodhidharma"

    for n_bars, batch_size in zip(n_bars, batch_sizes):
        logger.info("#" * 30)
        logger.info(f"Experiment: n_bars = {n_bars}")
        cur_config = load_config(config_path)

        # Update config file with experiment settings
        cur_config["preprocessing"]["n_bars"] = n_bars
        cur_config["CycleGAN"]["n_timesteps"] = n_bars * 16
        cur_config["training"]["batch_size"] = batch_size

        prep_path = f"tfrecord/{n_bars}_bars"

        logger.info(f"Saving {config_outfile}")
        with open(config_outfile, "w") as f:
            dump(cur_config, f)

        logger.info("Training with new config...")
        train_path_a = os.path.join("../data/cycleGAN", prep_path, genre_a)
        train_path_b = os.path.join("../data/cycleGAN", prep_path, genre_b)
        logger.debug(f"path_a: {train_path_a}")
        logger.debug(f"path_b: {train_path_b}")
        train_pipeline(
            [
                train_path_a,
                train_path_b,
                genre_a,
                genre_b,
                "--config_path",
                config_outfile,
            ]
        )
        logger.info("[Done]")

        model_name = get_latest_trained_model()
        model_fpath = f"trained_models/{model_name}"

        transfer_outpath = os.path.join(convert_outpath, f"{n_bars}_bars")
        logger.info(f"transfer_outpath: {transfer_outpath}")
        style_transfer(
            [
                os.path.join(bodhidharma_path, prep_path, bd_genre_a),
                os.path.join(bodhidharma_path, prep_path, bd_genre_b),
                bd_genre_a,
                bd_genre_b,
                model_fpath,
                "--outpath",
                transfer_outpath,
                "--config_fpath",
                "src/train_config.yaml",
            ]
        )

        eval_results_path = os.path.join(results_outpath, f"{n_bars}_bars")
        eval_experiment(
            [
                transfer_outpath,
                bd_genre_a,
                bd_genre_b,
                model_name,
                eval_results_path,
            ]
        )
        logger.info("")


if __name__ == "__main__":
    main(sys.argv[1:])
