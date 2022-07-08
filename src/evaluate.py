import os
import numpy as np
import pretty_midi
from glob import glob
import json
import logging

from eval_utils import (
    eval_chroma_similarities,
    gen_histograms,
    time_pitch_diff_hist,
    onset_duration_hist,
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

    return (original_songs, transfer_songs)


def compute_style_metric(
    name,
    tup_a,
    tup_b,
    hist_func,
    genre_a,
    genre_b,
    **kwargs,
):
    """

    Parameters
    ----------
    name : str
        Name of the metric calculation, to be used when writing the results.
        Input songs.
    tup_a : tuple(list[pretty_midi.PrettyMIDI], list[pretty_midi.PrettyMIDI])
        Tuple containing the original songs in style A and the songs transferred to style B.
    tup_b : tuple(list[pretty_midi.PrettyMIDI], list[pretty_midi.PrettyMIDI])
        Tuple containing the original songs in style B and the songs transferred to style A.
    hist_func : function
        Histogram metric to compute. One of (`time_pitch_diff_hist`. `onset_duration_hist`)
    genre_a : str
        Name of genre A.
    genre_b : str
        Name of genre B.
    kwargs :
        Keyword arguments to pass to `metric_func`.

    Returns
    -------
    List[np.array]
        The computes time-pitch histograms for each input song.
    """
    inputs_a, a_transfer_b = tup_a
    inputs_b, b_transfer_a = tup_b

    histograms_a = gen_histograms(inputs_a, hist_func=hist_func, **kwargs)
    histograms_a2b = gen_histograms(a_transfer_b, hist_func=hist_func, **kwargs)
    a_reference_hist = histograms_a.mean(axis=0)
    a2b_reference_hist = histograms_a2b.mean(axis=0)

    histograms_b = gen_histograms(inputs_b, hist_func=hist_func, **kwargs)
    histograms_b2a = gen_histograms(b_transfer_a, hist_func=hist_func, **kwargs)
    b_reference_hist = histograms_b.mean(axis=0)
    b2a_reference_hist = histograms_b2a.mean(axis=0)

    results = {
        f"macro_{name}": {
            f"{genre_a}2{genre_b}": eval_style_similarities(
                [a2b_reference_hist], b_reference_hist
            ),
            f"{genre_b}2{genre_b}": eval_style_similarities(
                [b2a_reference_hist], a_reference_hist
            ),
        },
        f"per_song_{name}": {
            f"{genre_a}2{genre_b}": eval_style_similarities(
                histograms_a2b, b_reference_hist
            ),
            f"{genre_b}2{genre_a}": eval_style_similarities(
                histograms_b2a, a_reference_hist
            ),
        },
    }
    return results


if __name__ == "__main__":
    chroma_args = dict(sampling_rate=12, window_size=24, stride=12, use_velocity=False)
    hist_kwargs = dict(max_time=4, bin_size=1 / 6, normed=True)

    # Test args
    base = "converted/{}/{}/*.mid*"
    model = "CP_C2CP_P_30e_bs32_nr84_ts64_sd1_run_2022_06_25-19_24_32"
    pattern_A2B = base.format(model, "A2B")
    pattern_B2A = base.format(model, "B2A")
    fpaths_A2B = glob(pattern_A2B)
    fpaths_B2A = glob(pattern_B2A)
    genre_a = "C"
    genre_b = "P"
    outpath = f"results/{model}"

    tup_a = load_converted_songs(fpaths_A2B)
    tup_b = load_converted_songs(fpaths_B2A)

    logger.info("Computing chroma_similarities...")
    results = {
        "chroma_similarities": {
            f"{genre_a}2{genre_b}": eval_chroma_similarities(*tup_a, **chroma_args),
            f"{genre_b}2{genre_a}": eval_chroma_similarities(*tup_b, **chroma_args),
        }
    }

    logger.info(f"Computing time-pitch histograms")
    time_pitch_results = compute_style_metric(
        "time_pitch_diff",
        tup_a,
        tup_b,
        time_pitch_diff_hist,
        genre_a,
        genre_b,
        **hist_kwargs,
    )
    results.update(time_pitch_results)

    logger.info(f"Computing onset-duration histograms")
    onset_duration_results = compute_style_metric(
        "onset_duration", tup_a, tup_b, onset_duration_hist, genre_a, genre_b
    )
    results.update(onset_duration_results)

    write_output(results, outpath, genre_a, genre_b)

    logger.info("Done!")
