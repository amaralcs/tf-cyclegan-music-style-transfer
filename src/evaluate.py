import os
import numpy as np
import pretty_midi
from glob import glob
import json
import logging

from eval_utils import (
    eval_chroma_similarities,
    gen_histograms,
    eval_style_similarities,
)

logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def write_output(results, outpath, genre_a, genre_b):
    os.makedirs(outpath, exist_ok=True)
    outfile = f"{outpath}/{genre_a}2{genre_b}_results.json"

    logger.info(f"Writing results to {outfile}")
    with open(outfile, "w") as f:
        json.dump(results, f)


def load_converted_songs(fpaths):
    """Loads the original and transferred songs from the given paths.

    Parameters
    ----------
    fpaths : List[str]
        Paths to the converted songs.
        The directory must contain files of the form:
            x_original.mid, x_transfer.mid and x_cycle.mid
        containing the original phrase, the result of the transfer and the cycle back to the original style.

    Returns
    -------
    List[prettymidi.PrettyMIDI], List[prettymidi.PrettyMIDI]
    """
    logger.info(f"Loading files from {os.path.split(fpaths[0])[0]}")
    # cycled_fpaths = [f for f in fpaths if "cycle.mid" in f]
    original_fpaths = [f for f in fpaths if "original.mid" in f]
    transfer_fpaths = [f for f in fpaths if "transfer.mid" in f]

    # cycled_songs = [pretty_midi.PrettyMIDI(filepath) for filepath in cycled_fpaths]
    original_songs = [pretty_midi.PrettyMIDI(filepath) for filepath in original_fpaths]
    transfer_songs = [pretty_midi.PrettyMIDI(filepath) for filepath in transfer_fpaths]

    return original_songs, transfer_songs


if __name__ == "__main__":
    chroma_args = dict(sampling_rate=12, window_size=24, stride=12, use_velocity=False)
    hist_kwargs = dict(max_time=4, bin_size=1 / 6, normed=True)

    # Debug args
    base = "converted/{}/{}/*.mid*"
    model = "CP_C2CP_P_30e_bs32_nr84_ts64_sd1_run_2022_06_25-19_24_32"
    pattern_A2B = base.format(model, "A2B")
    pattern_B2A = base.format(model, "B2A")
    fpaths_A2B = glob(pattern_A2B)
    fpaths_B2A = glob(pattern_B2A)
    genre_a = "C"
    genre_b = "P"
    outpath = f"results/{model}"

    inputs_a, a_transfer_b = load_converted_songs(fpaths_A2B)
    inputs_b, b_transfer_a = load_converted_songs(fpaths_B2A)

    logger.info("Computing chroma_similarities...")
    results = {
        "chroma_similarities": {
            "A2B": eval_chroma_similarities(inputs_a, a_transfer_b, **chroma_args),
            "B2A": eval_chroma_similarities(inputs_b, b_transfer_a, **chroma_args),
        },
        "agg_style_profile": {"A2B": None, "B2A": None},
        "per_song_style_profile": {"A2B": None, "B2A": None},
    }

    logger.info(f"Computing style histograms")
    histograms_a = gen_histograms(inputs_a, hist_kwargs)
    histograms_a2b = gen_histograms(a_transfer_b, hist_kwargs)
    a_ref_hist = histograms_a.mean(axis=0)
    a2b_ref_hist = histograms_a2b.mean(axis=0)

    histograms_b = gen_histograms(inputs_b, hist_kwargs)
    histograms_b2a = gen_histograms(b_transfer_a, hist_kwargs)
    b_ref_hist = histograms_a.mean(axis=0)
    b2a_ref_hist = histograms_b2a.mean(axis=0)

    results["agg_style_profile"]["A2B"] = eval_style_similarities(
        [a2b_ref_hist], b_ref_hist
    )
    results["agg_style_profile"]["B2A"] = eval_style_similarities(
        [b2a_ref_hist], a_ref_hist
    )
    results["per_song_style_profile"]["A2B"] = eval_style_similarities(
        histograms_a2b, b_ref_hist
    )

    results["per_song_style_profile"]["B2A"] = eval_style_similarities(
        histograms_b2a, a_ref_hist
    )

    write_output(results, outpath, genre_a, genre_b)

    logger.info("Done!")
