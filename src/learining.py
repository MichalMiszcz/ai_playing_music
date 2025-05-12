import mido
import subprocess
import os
import sys
from os import mkdir
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from music21 import converter
from pdf2image import convert_from_path
import shutil


# Step 2: Preprocess MIDI files (remove tempo and irregular groupings)
def remove_tempo_info(midi_file, output_midi_file):
    """
    Removes tempo (set_tempo) information from a MIDI file.
    """
    mid = mido.MidiFile(midi_file)
    for track in mid.tracks:
        track[:] = [msg for msg in track if msg.type != 'set_tempo']
    mid.save(output_midi_file)

def normalize_note_durations(midi_file, output_midi_file):
    """
    Normalizes note durations in a MIDI file to remove irregular groupings (e.g., tuplets).
    """
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        time = 0
        messages = []
        note_starts = {}
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_starts[msg.note] = (time, msg.velocity)
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                if msg.note in note_starts:
                    start_time, velocity = note_starts.pop(msg.note)
                    duration = time - start_time
                    messages.append((start_time, 'note_on', msg.note, velocity, duration))

        normalized_track = []
        last_time = 0
        for start_time, msg_type, note, velocity, duration in sorted(messages, key=lambda x: x[0]):
            duration_ticks = duration
            standard_duration = round(duration_ticks / (ticks_per_beat / 2)) * (ticks_per_beat / 2)
            if standard_duration == 0:
                standard_duration = ticks_per_beat / 2

            delta_time = start_time - last_time
            normalized_track.append(mido.Message('note_on', note=note, velocity=velocity, time=int(delta_time)))
            note_off_time = start_time + standard_duration
            normalized_track.append(mido.Message('note_off', note=note, velocity=0, time=int(standard_duration)))
            last_time = note_off_time

        new_track = []
        for msg in track:
            if msg.type not in ['note_on', 'note_off']:
                new_track.append(msg)
        new_track.extend(normalized_track)
        track[:] = sorted(new_track, key=lambda msg: msg.time)

    mid.save(output_midi_file)


# Process all MIDI files
midi_folder_path = "data/midi"
processed_midi_folder_path = "data/processed_midi"
if not os.path.exists(processed_midi_folder_path):
    os.makedirs(processed_midi_folder_path)

for root, dirs, files in os.walk(midi_folder_path):
    for folder in dirs:
        active_midi_folder = os.path.join(midi_folder_path, folder)
        active_processed_folder = os.path.join(processed_midi_folder_path, folder)
        if not os.path.exists(active_processed_folder):
            os.makedirs(active_processed_folder)

        for file in os.listdir(active_midi_folder):
            if file.endswith(".mid"):
                input_midi_file = os.path.join(active_midi_folder, file)
                temp_midi_file = os.path.join(active_processed_folder, file.replace(".mid", "_temp.mid"))
                output_midi_file = os.path.join(active_processed_folder, file)

                # Remove tempo and normalize durations
                remove_tempo_info(input_midi_file, temp_midi_file)
                # normalize_note_durations(temp_midi_file, output_midi_file)
                os.remove(temp_midi_file)
                print(f"Processed: {output_midi_file}")


