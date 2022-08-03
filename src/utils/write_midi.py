""" 
    This is the original `write_midi` file from the original paper repo.
    It could probably use some refactoring.
"""

import numpy as np
import pretty_midi


def set_piano_roll_to_instrument(
    piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=16
):
    # Calculate time per pixel
    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 4
    phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape(
        (piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2])
    )  # remove batch axis (from (1, 64, 128) -> (64, 128)))
    piano_roll_diff = np.concatenate(
        (np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int))
    )

    # np.diff calculates the diff between element x+1 and x for all x in an array
    piano_roll_search = np.diff(piano_roll_diff.round(), axis=0)

    # Next we'll iterate through all columns in the piano roll (remember, the piano roll is transposed)
    # So for each column (i.e. pitch) we find the start times where the notes start being played
    # and subtract these from the end times to calculate the duration of each note
    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))
        # print('start_time:', start_time)
        # print(len(start_time))
        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))
        # print('end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('duration each note:', duration)
        # print(len(duration))

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        # for onset in all_onsets
        for i in range(len(start_time)):
            # print(start_time)
            # if onset is in temp_onset_times and is not the last onset
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                # print('i and start_time:', i, start_time[i])
                t = []
                current_idx = temp_start_time.index(
                    start_time[i]
                )  # get the index of this onset
                for j in range(
                    current_idx + 1, len(temp_start_time)
                ):  # Then for the remaining elements of temp_onset_times
                    # print(j, temp_start_time[j])
                    # check that the next start time is smaller than current onset + a threshold
                    # and that the offset of the next note is <= than current onset + a threshold
                    if (
                        temp_start_time[j] < start_time[i] + threshold
                        and temp_end_time[j] <= start_time[i] + threshold
                    ):
                        # print('popped start time:', temp_start_time[j])
                        t.append(j)  # add it to a list of elements to remove
                        # print('popped temp_start_time:', t)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])
                # print('popped temp_start_time:', temp_start_time)

        start_time = temp_start_time
        # print('After checking, start_time:', start_time)
        # print(len(start_time))
        end_time = temp_end_time
        # print('After checking, end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('After checking, duration each note:', duration)
        # print(len(duration))

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        # Iterate through all the searched notes
        # creating a note object for each one, with pitch, start time, end time and the given velocity
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_num,
                    start=start_time[idx],
                    end=end_time[idx],
                )
                # Add the note to the Instrument object
                instrument.notes.append(note)
            else:
                # Check that we won't go out of bounds (i.e. note duration isn't longer than the song)
                if start_time[idx] + threshold <= phrase_end_time:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=note_num,
                        start=start_time[idx],
                        end=start_time[idx] + threshold,
                    )
                else:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=note_num,
                        start=start_time[idx],
                        end=phrase_end_time,
                    )
                # Add the note to the Instrument object
                instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
    # print(max([i.end for i in instrument.notes]))
    # print('tpp, threshold, phrases_end_time:', tpp, threshold, phrase_end_time)


def write_piano_roll_to_midi(
    piano_roll,
    filename,
    program_num=0,
    is_drum=False,
    velocity=100,
    tempo=120.0,
    beat_resolution=16,
):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # Set the piano roll to the Instrument object
    set_piano_roll_to_instrument(
        piano_roll, instrument, velocity, tempo, beat_resolution
    )
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


def write_piano_rolls_to_midi(
    piano_rolls,
    program_nums=None,
    is_drum=None,
    filename="test.mid",
    velocity=100,
    tempo=120.0,
    beat_resolution=24,
):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums have different sizes...")
        return False
    if not program_nums:
        program_nums = [0, 0, 0]
    if not is_drum:
        is_drum = [False, False, False]
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for idx in range(len(piano_rolls)):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(
            program=program_nums[idx], is_drum=is_drum[idx]
        )
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(
            piano_rolls[idx], instrument, velocity, tempo, beat_resolution
        )
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)
