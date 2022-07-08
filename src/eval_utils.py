"""
    The following script was adapted from:
        Cífka, O., Şimşekli, U. and Richard, G. (2019) 
        ‘Supervised symbolic music style translation using synthetic data’, in Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019. International Society for Music Information Retrieval, pp. 588–595. 
        doi: 10.5281/zenodo.3527878.

    Original implementation:
    https://github.com/cifkao/ismir2019-music-style-translation/tree/master/code/ismir2019_cifka/eval
"""
import numpy as np
import pretty_midi
from scipy.signal import convolve2d
from sklearn.metrics.pairwise import cosine_similarity


def _as_note_list(sequence):
    """Creates a list of notes from the input sequence.

    Parameters
    ----------
    sequence : List[prettymidi.PrettyMIDI]

    Returns
    -------
    List[prettymidi.Note]
    """
    if isinstance(sequence, pretty_midi.PrettyMIDI):
        sequence = [
            note for instrument in sequence.instruments for note in instrument.notes
        ]

    return sequence


def _strip_velocity(notes):
    """Defaults the velocity of all notes to 127.

    Parameters
    ----------
    notes : List[prettymidi.Note]

    Returns
    -------
    List[prettymidi.Note]
    """
    return [
        pretty_midi.Note(pitch=n.pitch, start=n.start, end=n.end, velocity=127)
        for n in notes
    ]


def _get_chroma(notes, sampling_rate):
    """Creates a midi track with a single instrument using the given inputs notes and
    computes the chromagram using the given sampling_rate.
    The default instrument used is the Grand Piano (midi index 0).

    Parameters
    ----------
    notes : List[prettymidi.Note]
        List of notes.
    sampling_rate : int
        Sampling frequency of the columns, i.e. each column is spaced apart by ``1./fs`` seconds.

    Returns
    -------
    np.array
        The resulting chromagram.
    """
    midi = pretty_midi.Instrument(0)  # Grand Piano
    midi.notes[:] = notes
    return midi.get_chroma(fs=sampling_rate)


def _average_cos_similarity(chroma_a, chroma_b):
    """Compute the column-wise cosine similarity, averaged over all non-zero columns.

    Parameters
    ----------
    chroma_a : np.array
        The convolved chromagram of input a.
    chroma_b : np.array
        The convolved chromagram of input b.

    Returns
    -------
    int
        The average cosine similarity over all non-zero columns.
    """
    nonzero_cols_ab = []
    for chroma in (chroma_a, chroma_b):
        col_norms = np.linalg.norm(chroma, axis=0)
        nonzero_cols = col_norms > 1e-9
        nonzero_cols_ab.append(nonzero_cols)

        # Note: 'np.divide' needs the 'out' parameter, otherwise the output would get written to
        # an uninitialized array.
        np.divide(chroma, col_norms, where=nonzero_cols, out=chroma)

    # Count the columns where at least one of the two matrices is nonzero.
    num_nonzero_cols = np.logical_or(*nonzero_cols_ab).sum()

    # Compute the dot product.
    return np.tensordot(chroma_a, chroma_b) / num_nonzero_cols


def _convolve_strided(data, filter_, stride):
    """Compute a 2D convolution with the given stride along the second dimension.

    A full (zero-padded) 2D convolution is computed, then subsampled according to the stride with
    an offset calculated so that the convolution window is aligned to the left edge of the original
    array.

    Parameters
    ----------
    data : np.array
        Chromagram to convolve.
    filter_ : np.array
        Convolution filter to use.
    stride : int
        The size of the strides.

    Returns
    -------
    np.array
        The strided convolution, aligned ot the left edfe of the original array.
    """
    convolution = convolve2d(data, filter_, mode="full")
    offset = (filter_.shape[-1] - 1) % stride  # Make sure the windows are aligned
    return convolution[:, offset::stride]


def chroma_similarity(
    sequence_a, sequence_b, sampling_rate, window_size, stride, use_velocity=False
):
    """Computes the average chroma similarity of the given inputs sequence over the given window_size
    using the given stride.

    Parameters
    ----------
    sequence_a : pretty_midi.PrettyMIDI
        First sequence, as a PrettyMIDI object.
    sequence_b : pretty_midi.PrettyMIDI
        Second sequence, as a PrettyMIDI object.
    sampling_rate : int
        Sampling frequency of the columns, i.e. each column is spaced apart by ``1./fs`` seconds.
    window_size : int
        The size of the convolution to apply to each chroma.
    stride: int
        The stride of the window used during the convolution.
    use_velocity : bool
        Whether to take into account the velocities or not.

    Returns
    -------
    float
        The cosine similarity of the chromas of sequence_a and sequence_b.
    """
    notes_a, notes_b = (_as_note_list(seq) for seq in (sequence_a, sequence_b))

    if not use_velocity:
        notes_a, notes_b = (_strip_velocity(notes) for notes in (notes_a, notes_b))

    chroma_a, chroma_b = (
        _get_chroma(notes, sampling_rate) for notes in (notes_a, notes_b)
    )

    # Make sure the chroma matrices have the same dimensions.
    if chroma_a.shape[1] < chroma_b.shape[1]:
        chroma_a, chroma_b = chroma_b, chroma_a
    pad_size = chroma_a.shape[1] - chroma_b.shape[1]
    chroma_b = np.pad(chroma_b, [(0, 0), (0, pad_size)], mode="constant")

    # Compute a moving average over time.
    avg_filter = np.ones((1, window_size)) / window_size
    chroma_avg_a, chroma_avg_b = (
        _convolve_strided(chroma, avg_filter, stride) for chroma in (chroma_a, chroma_b)
    )

    return _average_cos_similarity(chroma_avg_a, chroma_avg_b)


def eval_chroma_similarities(inputs_, transfers, **kwargs):
    """Compute the chroma similarity between the given inputs and the transfer to the target genre.

    Parameters
    ----------
    inputs_ : List[prettymidi.PrettyMIDI]
    transfer : List[prettymidi.PrettyMIDI]
    **kwargs :
        Arguments to be passed to `chroma_similarity`

    Return
    ------
    List[float]
        The computed chroma similarities.
    """
    return [
        chroma_similarity(original, transfer, **kwargs)
        for original, transfer in zip(inputs_, transfers)
    ]


def time_pitch_diff_hist(
    sequences, max_time=2, bin_size=1 / 6, pitch_range=20, normed=False
):
    """Compute an onset-time-difference vs. interval histogram.

    Parameters
    ----------
    sequences : List[List[pretty_midi.Note]]
        A list of notes for each instrument.
    max_time : int
        The maximum time between two notes to be considered.
    bin_size : float
        The bin size along the time axis.
    pitch_range : int
        The number of pitch difference bins in each direction (positive or negative, excluding 0).
        Each bin has size 1.
    normed : bool
        Whether to normalize the histogram.

    Returns
    -------
    np.array
        A 2D histogram of shape `[max_time / bin_size, 2 * pitch_range + 1]`.
    """
    epsilon = 1e-9
    time_diffs = []
    intervals = []
    for notes in sequences:
        onsets = [n.start for n in notes]

        # Compute the deltas of all onsets
        diff_mat = np.subtract.outer(onsets, onsets)

        # Only keep positive time differences.
        index_pairs = zip(
            *np.where((diff_mat < max_time - epsilon) & (diff_mat >= 0.0))
        )
        for j, i in index_pairs:
            # Do not count the difference against itself
            if j != i:
                time_diffs.append(diff_mat[j, i])
                intervals.append(notes[j].pitch - notes[i].pitch)

    histogram, _, _ = np.histogram2d(
        intervals,
        time_diffs,
        normed=normed,
        bins=[
            np.arange(-(pitch_range + 1), pitch_range + 1) + 0.5,
            np.arange(0.0, max_time + bin_size - epsilon, bin_size),
        ],
    )
    return np.nan_to_num(histogram)


def onset_duration_hist(
    sequences, bar_duration=4, max_value=2, bin_size=1 / 6, normed=True
):
    """Compute an onset-duration histogram.

    The x-axis of the histogram contains the note durations while the y-axis contains the
    position of the note onset relative to the beat (considering the given `bar_duration`).

    Parameters
    ----------
    sequences : List[List[pretty_midi.Note]]
        A list of notes for each instrument.
    bar_duration : int
        The number of beats per bar.
    max_value : int
        The maximum distance between an onset and offset.
    bin_size : float
        The bin size along the time axis.
    normed : bool
        Whether to normalize the histogram.

    Returns
    -------
    np.array
        A 2D histogram of shape `[bar_duration / bin_size, max_value / bin_size]`.

    Notes
    -----
    This implementation does not support sequences with more than 1 track. Note that the inputs
    to `histogram2D` take only the first entry of the lists. For a more complete implementation one may
    want to explore using `np.histogramdd`.
    """
    epsilon = 1e-9
    onsets = []
    durations = []
    for notes in sequences:
        note_onsets = [n.start % bar_duration for n in notes]
        note_durations = [n.end - n.start for n in notes]

        onsets.append(note_onsets)
        durations.append(note_durations)

    histogram, _, _ = np.histogram2d(
        onsets[0],
        durations[0],
        normed=normed,
        bins=[
            np.arange(0.0, bar_duration + bin_size - epsilon, bin_size),
            np.arange(0.0, max_value + bin_size - epsilon, bin_size),
        ],
    )
    return np.nan_to_num(histogram)


def gen_histograms(sequences, hist_func, **kwargs):
    """Helper function to compute the time_pitch histogram of a given sequence of songs.

    Parameters
    ----------
    sequences : List[prettymidi.PrettyMIDI]
        Input songs.
    hist_func : function
        Histogram metric to compute. One of (`time_pitch_diff_hist`. `onset_duration_hist`)
    kwargs :
        Keyword arguments to pass to `metric_func`.

    Returns
    -------
    List[np.array]
        The computes time-pitch histograms for each input song.
    """
    hists = []
    for sequence in sequences:
        notes = [note for instr in sequence.instruments for note in instr.notes]
        hists.append(hist_func([notes], **kwargs))
    return np.array(hists)


def eval_style_similarities(histograms, ref_histogram):
    """Computes the cosine similarity between a list of input histograms and a reference histogram.

    Histograms are computed using `gen_histogram`, `ref_histogram` should be an
    average of all histograms computed for a given genre.

    Parameters
    ----------
    histograms : List[np.array]
        Style fit histograms for the input songs
    ref_histogram : np.array
        Average histogram of a given genre.

    Returns
    -------
    List[float]
        The cosine similarities of the given histograms against a reference histogram.
    """
    ref_H = ref_histogram.ravel()

    similarities = []
    for H in histograms:
        H = H.ravel()
        [[similarity]] = cosine_similarity([H], [ref_H])
        similarities.append(similarity)

    return similarities
