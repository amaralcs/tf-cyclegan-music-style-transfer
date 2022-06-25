"""
    Prepares and processes a MIDI dataset.

    Steps include:
        - Filtering tracks
        - Removing velocity
        - Limiting tracks to a specified range of notes
        - Train/test split
        - Splitting tracks to individual phrases
        - Saving phrases as tfrecord tensors

    Usage
    -----
    python src/prepare.py new_dataset pop \
        --outpath tfrecord
            
    Note
    ----
    We expect the data to be processed to be in a folder of name `genre` located
    in `root_path`. (e.g. Pop midi tracks should be located in new_dataset/pop)
"""

import logging
import os
import errno
import sys
import numpy as np
from argparse import ArgumentParser

from tensorflow.io import serialize_tensor, TFRecordWriter

from pypianoroll import Multitrack, Track, from_pretty_midi
import pretty_midi


logger = logging.getLogger("preprocessing_logger")
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
    args.add_argument(
        "root_path",
        default="processed_midi",
        type=str,
        help="Path to where the midis are stored.",
    )
    args.add_argument(
        "genre", default="pop", type=str, help="Genre of the midis being processes."
    )
    args.add_argument(
        "--outpath",
        default="tfrecord",
        type=str,
        help="Name of directory with outputs.",
    )
    args.add_argument(
        "--test-ratio",
        default=0.3,
        type=float,
        help="Ratio of files to use for testing.",
    )
    args.add_argument(
        "--remove-velocity",
        default=True,
        type=bool,
        help="Whether to remove velocity information from the final outputs.",
    )
    args.add_argument(
        "--clip-low",
        default=24,
        type=int,
        help="Lower bound for notes to use (0 - 128 in a midi file)",
    )
    args.add_argument(
        "--clip-high",
        default=108,
        type=int,
        help="Upper bound for notes to use (0 - 128 in a midi file)",
    )
    args.add_argument(
        "--drop-phrases",
        default=True,
        type=bool,
        help=(
            "If set to true, only keep tracks that have 4 bars or keep a single phrase of "
            "tracks that do not have number of tracks a multiple of 4."
        ),
    )
    return args.parse_args(argv)


def get_midi_path(root):
    """Return a list of paths to MIDI files in `root` (recursively)

    Parameters
    ----------
    root : str
        Path to directory with midi files.

    Returns
    -------
    List[str]
    """
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if ".mid" in filename:
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths


def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not exist.

    Parameters
    ----------
    path : str
        Name of folder to create.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise exception


def get_midi_info(midi_obj):
    """Return stats of the loaded midi object.

    The particular stats we are interested in are:
        - first_beat_time
        - num_time_signature_changes (was there changes in the time signature?)
        - time_signature (None, if there were changes in the signature)
        - tempo

    Parameters
    ----------
    midi_obj : pretty_midi.PrettyMIDI
        The pretty middy object

    Returns
    -------
    dict
    """

    if midi_obj.time_signature_changes:
        # if there was change, take the first beat time from the earliest signature
        midi_obj.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = midi_obj.time_signature_changes[0].time
    else:
        first_beat_time = midi_obj.estimate_beat_start()

    tc_times, tempi = midi_obj.get_tempo_changes()

    if len(midi_obj.time_signature_changes) == 1:
        time_sign = "{}/{}".format(
            midi_obj.time_signature_changes[0].numerator,
            midi_obj.time_signature_changes[0].denominator,
        )
    else:
        time_sign = None

    midi_info = {
        "first_beat_time": first_beat_time,
        "num_time_signature_change": len(midi_obj.time_signature_changes),
        "time_signature": time_sign,
        "tempo": tempi[0] if len(tc_times) == 1 else None,
    }

    return midi_info


def get_merged_multitrack(multitrack):
    """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
    five tracks (Bass, Drums, Guitar, Piano and Strings)

    Parameters
    ----------
    multitrack : pypianoroll.Multitrack

    Returns
    -------
    pypianoroll.Multitrack
    """
    category_list = {"Bass": [], "Drums": [], "Guitar": [], "Piano": [], "Strings": []}
    program_dict = {"Piano": 0, "Drums": 0, "Guitar": 24, "Bass": 32, "Strings": 48}

    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            category_list["Drums"].append(idx)
        elif track.program // 8 == 0:
            category_list["Piano"].append(idx)
        elif track.program // 8 == 3:
            category_list["Guitar"].append(idx)
        elif track.program // 8 == 4:
            category_list["Bass"].append(idx)
        else:
            category_list["Strings"].append(idx)

    tracks = []
    for key in category_list:
        if category_list[key]:
            pianoroll = Multitrack(
                tracks=[multitrack[i] for i in category_list[key]]
            ).blend()
        else:
            pianoroll = None
        tracks.append(
            Track(
                pianoroll=pianoroll,
                program=program_dict[key],
                is_drum=key == "Drums",
                name=key,
            )
        )
    return Multitrack(
        tracks=tracks,
        tempo=multitrack.tempo,
        downbeat=multitrack.downbeat,
        resolution=multitrack.resolution,
        name=multitrack.name,
    )


def create_multitracks(filepath):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`

    Parameters
    ----------
    filepath : str
        Path to midi file to be converted.
    multitrack_path : str
        Path to save outputs as .npz files.

    Returns
    -------
    List[str, dict]
    """
    midi_name = os.path.splitext(os.path.basename(filepath))[0]

    try:
        # Create the multitrack object
        multitrack = Multitrack(resolution=24, name=midi_name)

        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = get_midi_info(pm)

        # This adds the pianoroll to the multitrack
        multitrack = from_pretty_midi(pm)
        merged = get_merged_multitrack(multitrack)

        return [midi_name, midi_info, merged]

    except TypeError:
        print(f"There was a type error when loading {midi_name}")
        return None


def filter_track(
    midi_info, beat_time=0, time_signature_changes=1, allowed_signatures=["4/4"]
):
    """Returns True if the track is to be filtered, else returns False.

    Files are qualified based on `first_beat_time`, `num_time_signature_change` and
    `time_signature`.
    If you do not wish to filter for time signatures pass None to `allowed_signatures`.

    Returns
    -------
    boolean
    """
    if midi_info["first_beat_time"] > beat_time:
        return True
    elif midi_info["num_time_signature_change"] > time_signature_changes:
        return True
    elif allowed_signatures:
        if midi_info["time_signature"] not in allowed_signatures:
            return True
    return False


def convert_and_clean_midis(midi_fpath, **filter_kwargs):
    """Loads the midis selected in the first step and converts them to pypianoroll.Multitrack and filters
    based on the given rules.

    Creating the files as multitrack also merges instruments into a single track.

    Parameters
    ----------
    midi_fpath : str
        Path to midi files.
    filtered_fpath : str
        Output path for cleaned file.
    multitrack_path : st
        Output path for multitrack file.
    filter_kwargs : dict
        Options for filtering MIDI files.
    **filter_kwargs,

    Returns
    -------
    None
    """
    # Create full path

    # First step: load midis, create multitracks of each and save them
    midi_paths = get_midi_path(midi_fpath)
    logger.info(f"Found {len(midi_paths)} midi files")

    midi_tracks = [create_multitracks(midi_path) for midi_path in midi_paths]

    filtered_tracks = []
    for (name, info, track) in midi_tracks:
        if not filter_track(info, **filter_kwargs):
            track.name = name
            filtered_tracks.append(track)

    logger.info(f"[Done] {len(filtered_tracks)} have been kept.")
    return filtered_tracks


def shape_last_bar(piano_roll, last_bar_mode="remove"):
    """Utility function to handle the last bar of the song and reshape it.

    If `last_bar_mode` == "fill" then fill the remaining time steps with zeros.
    If `last_bar_mode` == "remove" then remove the remaining time steps.

    Parameters
    ----------
    piano_roll : np.array
        Piano roll array to trim
    """
    if int(piano_roll.shape[0] % 64) != 0:
        # Check that the number of time_steps is multiple of 4*16 (bars* required_time_steps)
        if last_bar_mode == "fill":
            piano_roll = np.concatenate(
                (piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0
            )

        elif last_bar_mode == "remove":
            piano_roll = np.delete(
                piano_roll, np.s_[-int(piano_roll.shape[0] % 64) :], axis=0
            )

    # Reshape to (-1, n_bars * n_time_steps, midi_range (128 notes))
    return piano_roll.reshape(-1, 64, 128)


def trim_midi_files(multitracks, clip_range=(24, 108), drop_phrases=True):
    """

    Parameters
    ----------
    multitracks : List
        List containing the loaded and filtered tracks.
    clip_range : tuple(int, int), Optional
        2-Tuple containing indices of midi range to clip. Use None if no clipping is desired.
    drop_phrases : True, Optional
        If set to true, only keep tracks that have 4 bars or keep a single phrase of tracks that do not
        have number of tracks a multiple of 4.

    Returns
    -------
    List
        A list with the trimmed files
    """
    trimmed_midis = []
    logger.info(f"Trimming {len(multitracks)} midi files...")

    for multitrack in multitracks:
        # A dict to keep track of index of instruments in the tracks
        track_indices = {"Piano": [], "Drums": []}
        logger.debug(f"\ttrimming {multitrack.name}")
        try:
            # Find the indices of Piano and Drums
            for idx, track in enumerate(multitrack.tracks):
                if track.is_drum:
                    track_indices["Drums"].append(idx)
                else:
                    track_indices["Piano"].append(idx)

            # Blend all piano tracks into a single track
            blended_multitrack = Multitrack(
                tracks=[multitrack[idx] for idx in track_indices["Piano"]]
            ).blend()

            shaped_multitrack = shape_last_bar(blended_multitrack)

            # Clip the range of the instruments to the indices given
            if clip_range:
                low, high = clip_range
                shaped_multitrack = shaped_multitrack[:, :, low:high]

            # Either keep all bars, or a single phrase
            if drop_phrases and (shaped_multitrack.shape[0] % 4 != 0):
                drop_bars_idx = int(shaped_multitrack.shape[0] % 4)
                shaped_multitrack = np.delete(
                    shaped_multitrack, np.s_[-drop_bars_idx:], axis=0
                )

            # Reshaped into (batchsize, 64, clipped_range, 1)
            shaped_multitrack = shaped_multitrack.reshape(-1, 64, 84, 1)

            trimmed_midis.append(shaped_multitrack)
        except Exception as err:
            raise err
    logger.info("[Done]")
    return trimmed_midis


def train_test_split(tracks, test_ratio):
    """Splits the files in a given directory into training and test sets

    Parameters
    ----------
    root_path : str
        Path to where the midis are stored.
    genre : str
        Genre of songs being processed. Should match a directory name in the `root_path`.
    test_ratio:
        Ratio of files used for testing the model.
    dataset : str
        Type of dataset being created. One of ["train", "test"]

    Returns
    -------
    train_set, test_set
    """
    test_indices = np.random.choice(
        len(tracks), int(test_ratio * len(tracks)), replace=False
    )
    train_indices = list(set(range(len(tracks))).difference(set(test_indices)))
    logger.info(
        f"Using {len(train_indices)} files for training and {len(test_indices)} for testing."
    )

    # Convert to np array for easy slicing.
    # Adding dtype="object" prevents a warning
    tracks = np.array(tracks, dtype="object")

    train_set = tracks[train_indices]
    test_set = tracks[test_indices]

    return train_set, test_set


def save(trimmed_midis, dataset, outpath, genre, remove_velocity):
    """Saves the processed midis as tfrecords.

    Parameters
    ----------
    trimmed_midis : List[np.array]
        List of midi files to convert.
    dataset : str
        Name of dataset to save (train or test)
    outpath : str
        Location to save the phrases.
    genre : str
        The name of the genre.
    remove_velocity : Boolean, Optional
        Whether to remove velocity information from the outputs.
    """
    outpath = os.path.join(outpath, genre, dataset)
    os.makedirs(outpath, exist_ok=True)

    # Aggregate all np arrays into a single array for ease of handling
    concat_midis = np.concatenate(trimmed_midis, axis=0)

    # Convert to an array of booleans if we want to omit velocity
    if remove_velocity:
        concat_midis = concat_midis > 0

    concat_midis = concat_midis.astype(np.float32)

    logger.info(f"Saving phrases to {outpath}")
    for idx, np_arr in enumerate(concat_midis):
        fname = os.path.join(outpath, f"{genre}_{idx+1}.tfrecord")

        tensor = serialize_tensor(np_arr)

        with TFRecordWriter(fname) as writer:
            writer.write(tensor.numpy())

    logger.info(f"[Done]")


def main(argv):
    """Main function to run the job."""
    args = parse_args(argv)
    root_path = args.root_path
    outpath = args.outpath
    genre = args.genre
    test_ratio = args.test_ratio
    remove_velocity = args.remove_velocity
    clip_range = (args.clip_low, args.clip_high)
    drop_phrases = args.drop_phrases

    midi_fpaths = os.path.join(root_path, genre)
    tracks = convert_and_clean_midis(midi_fpaths)
    trimmed_tracks = trim_midi_files(
        tracks, clip_range=clip_range, drop_phrases=drop_phrases
    )
    train_set, test_set = train_test_split(tracks=trimmed_tracks, test_ratio=test_ratio)

    outpath = os.path.join(root_path, outpath)

    save(train_set, "train", outpath, genre, remove_velocity)
    save(test_set, "test", outpath, genre, remove_velocity)


if __name__ == "__main__":
    main(sys.argv[1:])
