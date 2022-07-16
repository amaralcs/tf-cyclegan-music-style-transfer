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

logger = logging.getLogger("experiment_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s : %(name)s [%(levelname)s] : %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    settings = {"n_bars": [4]}
    config_outfile = "src/train_config.yaml"

    bodhidharma_path = "../data/bodhidharma"
    convert_outpath = "converted/bodhidharma"

    results_outpath = "results/bodhidharma"

    for n_bars in settings["n_bars"]:
        logger.info(f"Experiment: n_bars = {n_bars}")
        cur_config = load_config(config_path)

        # Update config file with experiment settings
        cur_config["preprocessing"]["n_bars"] = n_bars
        cur_config["CycleGAN"]["n_timesteps"] = n_bars * 16

        prep_outpath = f"prep_data/tfrecord/{n_bars}_bars"
        prep_data(
            [root_path, genre_a, "--n_bars", str(n_bars), "--outpath", prep_outpath]
        )
        prep_data(
            [root_path, genre_b, "--n_bars", str(n_bars), "--outpath", prep_outpath]
        )

        logger.info(f"Saving {config_outfile}")
        with open(config_outfile, "w") as f:
            dump(cur_config, f)

        logger.info("Training with new config...")
        train_path_a = os.path.join("../data/cycleGAN", prep_outpath, genre_a)
        train_path_b = os.path.join("../data/cycleGAN", prep_outpath, genre_b)
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

        # Prep bodhidharma data for transfer
        bd_prep_outpath = f"tfrecord/{n_bars}_bars"
        prep_data(
            [
                bodhidharma_path,
                bd_genre_a,
                "--n_bars",
                str(n_bars),
                "--test-ratio",
                "1",
                "--outpath",
                bd_prep_outpath,
            ]
        )
        prep_data(
            [
                bodhidharma_path,
                bd_genre_b,
                "--n_bars",
                str(n_bars),
                "--test-ratio",
                "1",
                "--outpath",
                bd_prep_outpath,
            ]
        )
        transfer_outpath = os.path.join(convert_outpath, f"{n_bars}_bars")
        logger.info(f"transfer_outpath: {transfer_outpath}")
        style_transfer(
            [
                os.path.join(bodhidharma_path, bd_prep_outpath, bd_genre_a),
                os.path.join(bodhidharma_path, bd_prep_outpath, bd_genre_b),
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


if __name__ == "__main__":
    main(sys.argv[1:])
