import sys
import os
import time
from argparse import ArgumentParser
from yaml import safe_load, YAMLError

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

from utils import load_data
from cyclegan import CycleGAN


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
    args.add_argument(
        "--batch_size", default=32, type=int, help="Batch size used for training."
    )
    args.add_argument(
        "--epochs", default=5, type=int, help="Number of epochs to train."
    )
    args.add_argument(
        "--model_output",
        default="trained_models",
        type=str,
        help="Name of directory to output model.",
    )
    args.add_argument(
        "--config_path",
        default="src/config.yaml",
        type=str,
        help="Path to YAML file containing model configuration.",
    )
    args.add_argument(
        "--log_dir",
        default="train_logs",
        type=str,
        help="Name of directory to output training logs.",
    )
    args.add_argument(
        "--learning_rate", default=0.0002, type=float, help="Initial Learning Rate"
    )
    args.add_argument(
        "--lr_step",
        default=3,
        type=int,
        help="Number of epochs before decreasing learning rate.",
    )
    args.add_argument(
        "--beta_1", default=0.5, type=float, help="Beta1 parameter for Adam Optimizer."
    )
    return args.parse_args(argv)


def lr_function_wrapper(lr, epochs, step):
    """Helper function to initialize the variable_lr function.

    Parameters
    ----------
    lr : float
        The initial learning rate.
    epochs : int
        The number of epochs the model will be trained for.
    step : int
        The number of epochs to maintain the initial lr.

    Returns
    -------
    function
    """

    def variable_lr(epoch):
        """Defines a variable learning rate.

        Parameters
        ----------
        epoch : int
            The current training epoch.

        Returns
        -------
        float
            The new learning rate.
        """
        if epoch < step:
            new_lr = lr
        else:
            new_lr = lr * (epochs - epoch) / (epochs - step)

        tf.summary.scalar("learning_rate", data=new_lr, step=epoch)
        return new_lr

    return variable_lr


def get_run_logdir(root_logdir, genre_a, genre_b, epochs, batch_size):
    """Generates the paths where the logs for this run will be saved.

    Parameters
    ----------
    root_logdir : str
        The base path to use.
    genre_a : str
        The name of genre A.
    genre_b : str
        The name of genre B.
    epochs : int
        Number of epochs the model will be trained for.
    batch_size : int
        The batch size used.

    Returns
    -------
    str, str
        The full path to the logging directory as well as the name of the current run.
    """
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    model_info = "{}2{}_{}e_bs{}_{}".format(
        genre_a, genre_b, epochs, batch_size, run_id
    )
    return os.path.join(root_logdir, model_info), model_info


def setup(log_dir, genre_a, genre_b, epochs, batch_size, learning_rate, step):
    """Creates the logging info, model name and defines the callbacks

    Parameters
    ----------
    log_dir : str
        Name of directory to log training details.
    genre_a : str
        Name of genre A.
    genre_b : str
        Name of genre B.
    epochs : int
        Number of epochs to train.
    batch_size : int
        Batch size to use for training.
    learning_rate : float
        The model initial learning rate.
    step : int
        Number of epochs before start decreasing the learning rate.

    Returns
    -------
    model_info : str
        String containing model training info.
    callbacks : List[tf.keras.callbacks]
        Callbacks to use during training.
    """
    run_logdir, model_info = get_run_logdir(
        log_dir, genre_a, genre_b, epochs, batch_size
    )
    file_writer = tf.summary.create_file_writer(run_logdir + "/metrics")
    file_writer.set_as_default()
    lr_function = lr_function_wrapper(learning_rate, epochs, step)

    callbacks = [LearningRateScheduler(lr_function), TensorBoard(log_dir=run_logdir)]
    return model_info, callbacks


def load_config(config_path):
    """Loads the model config from the given path

    Parameters
    ----------
    config_path : str
        Path to yaml file containing model configuration parameters.
    """
    with open(config_path, "r") as config_file:
        try:
            model_config = safe_load(config_file)
        except YAMLError as e:
            raise e
    return model_config["CycleGAN"]


def main(argv):
    """The main function for training.

    Note that this script only takes arguments related to the training. To change the model architecture,
    change the settings in config.yaml
    """
    args = parse_args(argv)
    path_a = args.path_a
    path_b = args.path_b
    model_output = args.model_output
    config_path = args.config_path
    log_dir = args.log_dir

    # debug path
    path_a = "data/JC_C_cp/tfrecord"  # dummy dir with less data
    path_b = "data/JC_J_cp/tfrecord"

    learning_rate = args.learning_rate
    step = args.lr_step
    beta_1 = args.beta_1
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer_params = dict(learning_rate=learning_rate, beta_1=beta_1)

    # TODO: Handle this a bit more cleanly
    genre_a = path_a.split("/")[1]
    genre_b = path_b.split("/")[1]
    if genre_a == path_a or genre_b == path_b:
        raise Exception(
            "There was an issue parsing the genre from the filename. Check the separator used."
        )

    os.makedirs(model_output, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = load_data(
        path_a, path_b, "train", batch_size=batch_size, cycle_length=500, shuffle=True
    )

    # Setup monitoring and callbacks
    model_info, callbacks = setup(
        log_dir, genre_a, genre_b, epochs, batch_size, learning_rate, step
    )

    # Setup model
    model_config = load_config(config_path)
    model = CycleGAN(genre_a, genre_b, **model_config)
    model.build_model(default_init=optimizer_params)

    model.fit(dataset, epochs=epochs, callbacks=callbacks)
    model.save_weights(f"{model_output}/{model_info}/weights/")


if __name__ == "__main__":
    main(sys.argv[1:])
