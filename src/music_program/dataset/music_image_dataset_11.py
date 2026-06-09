"""
Implementacja klasy datasetu. Dane uczące to obrazy zawierające tylko jedną pięciolinię oraz dwuwymiarowe MIDI.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import mido
from torchvision.transforms.v2.functional import to_pil_image

from src.music_program.utils.global_variables import *
from src.utils import index_to_note_delta_time
from src.utils.midi_sequence_conversion import normalize_3d_midi_sequence, create_2d_time_series

from collections import Counter

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}
velocity_to_index = {midi_num: i for i, midi_num in enumerate(VELOCITY)}
delta_time_to_index = {midi_num: i for i, midi_num in enumerate(DELTA_TIME)}


class MusicImageDataset(Dataset):
    def __init__(self, image_root, midi_root, left_hand_tracks=["Piano left"], right_hand_tracks=["Piano right"],
                 image_transform=None, max_seq_len=100, max_series_len=500, max_midi_files=100, modify_image=False,
                 aug_prob=0.0, learning=True):
        self.learning = learning

        self.image_root = image_root
        self.midi_root = midi_root
        self.left_hand_tracks = left_hand_tracks
        self.right_hand_tracks = right_hand_tracks
        self.image_transform = image_transform if image_transform else transforms.ToTensor()

        self.max_midi_duration = 4 * DELTA_TIME[NUM_DELTA_TIME - 1]
        self.max_seq_len = max_seq_len
        self.max_series_len = max_series_len
        self.modify_image = modify_image
        self.aug_prob = aug_prob

        midi_files = []
        for root, dirs, files in os.walk(midi_root):
            folder = os.path.basename(os.path.dirname(root))
            author = os.path.basename(root)
            for file in files:
                if file.endswith('.mid'):
                    midi_files.append((folder, author, os.path.join(root, file)))

        random.shuffle(midi_files)
        sorted_midi_files = sorted(midi_files[:max_midi_files], key=lambda x: x[2])
        self.selected_midi_files = sorted_midi_files

        self.image_paths = []
        self.midi_features = {}
        self.midi_time_seq = {}
        self.staff = {}
        records_to_remove = []

        self.lengths_of_midis = []

        for folder, author, midi_file in self.selected_midi_files:
            midi_name = os.path.splitext(os.path.basename(midi_file))[0]
            midi_key = f"{author}/{midi_name}"
            try:
                if self.learning:
                    staff_num = random.randint(1, 2)
                else:
                    staff_num = "all"

                try:
                    midi_seq = extract_notes_from_midi(midi_file, self.left_hand_tracks, self.right_hand_tracks,
                                                       self.max_midi_duration, staff_num)
                except Exception as e:
                    staff_num = 1
                    midi_seq = extract_notes_from_midi(midi_file, self.left_hand_tracks, self.right_hand_tracks,
                                                       self.max_midi_duration, staff_num)

                self.staff[midi_key] = staff_num

                if midi_seq is None:
                    print("Record to remove: ", folder, author, midi_file)
                    records_to_remove.append((folder, author, midi_file))
                    continue

                if len(midi_seq) > self.max_seq_len:
                    midi_seq = midi_seq[:self.max_seq_len]
                # else:
                #     midi_seq += [(0, 0, 0)] * (self.max_seq_len - len(midi_seq))

                # if len(midi_seq) > 32:
                #     raise Exception

                # normalized_seq = normalize_3d_midi_sequence(midi_seq)
                self.midi_features[midi_key] = midi_seq

                if learning:
                    max_len = self.max_series_len
                else:
                    max_len = self.max_seq_len

                self.midi_time_seq[midi_key] = create_2d_time_series(midi_seq, max_len, self.learning)

                # self.lengths_of_midis.append(len(self.midi_time_seq[midi_key]))

            except Exception as e:
                # print(f"Error processing MIDI {midi_file}: {e}")
                records_to_remove.append((folder, author, midi_file))
                continue

            folder = os.path.splitext(os.path.basename(midi_file))[0]
            file = os.path.splitext(os.path.basename(midi_file))[0] + "-1"  # Added only for my files
            image_dir = os.path.join(image_root, author, folder)

            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
                for file in image_files:
                    self.image_paths.append(os.path.join(image_dir, file))

        for record in records_to_remove:
            self.selected_midi_files.remove(record)

        # all_delta_times = [delta_time for seq in self.midi_features.values() for _, _, delta_time in seq if delta_time > 0]
        # self.max_delta_time = max(all_delta_times) if all_delta_times else 1
        self.max_delta_time = max(DELTA_TIME)

        self.image_paths.sort()
        print(f"Selected {len(self.selected_midi_files)} MIDI files and {len(self.image_paths)} images.")

        if len(self.image_paths) == 0:
            raise ValueError("No images found for the selected MIDI files. Check directory paths and file structure.")

        # counter = Counter(self.lengths_of_midis)
        # print(counter)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        rel_path = os.path.relpath(img_path, self.image_root)
        composer, piece, _ = rel_path.split(os.sep)
        midi_key = f"{composer}/{piece}"

        image = Image.open(img_path).convert('L')
        # image.show()

        if self.image_transform:
            image = self.image_transform(image)
            img_height = image.shape[1]
            part_to_cut = int(img_height / 3)
            part_to_cut_2 = int(2 * img_height / 3) - 1

            if self.learning is True:
                if self.staff[midi_key] == 1:
                    image = image[:, 0:part_to_cut, :]  # cutting first staff
                elif self.staff[midi_key] == 2:
                    image = image[:, part_to_cut:part_to_cut_2, :]  # cutting second staff
            # image_to_show = to_pil_image(image)
            # image_to_show.show("Modified image")

        normalized_seq = self.midi_time_seq.get(midi_key)
        midi_tensor_series = torch.tensor(normalized_seq, dtype=torch.long)

        return image, midi_tensor_series


def extract_notes_from_midi(midi_path, left_hand_tracks, right_hand_tracks, max_midi_duration, staff_num):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None

    left_hand_track = None
    right_hand_track = None

    for track in mid.tracks:
        if track.name in left_hand_tracks and left_hand_track is None:
            left_hand_track = track
        elif track.name in right_hand_tracks and right_hand_track is None:
            right_hand_track = track

    if left_hand_track is None and right_hand_track is None:
        print(f"MIDI file {midi_path} does not have the required tracks.")
        return None

    def extract_events_from_track(track, hand, staff_num):
        events = []
        current_time_idx = 0
        current_time = 0
        for msg in track:
            current_time += msg.time
            current_time_idx += delta_time_to_index[msg.time]

            def process_msg(msg):
                note_index = note_to_index[msg.note]
                note_velocity = velocity_to_index[msg.velocity] if msg.type == 'note_on' else 0

                return note_index, note_velocity

            if msg.type in ('note_on', 'note_off'):
                if msg.note in note_to_index:
                    if staff_num == "all":
                        note_idx, notes_velocity = process_msg(msg)
                        events.append((current_time_idx, note_idx, notes_velocity))
                    if staff_num == 1:
                        if current_time <= max_midi_duration:
                            note_idx, notes_velocity = process_msg(msg)
                            events.append((current_time_idx, note_idx, notes_velocity))
                    elif staff_num == 2:
                        if max_midi_duration <= current_time <= 2 * max_midi_duration:
                            note_idx, notes_velocity = process_msg(msg)
                            events.append((current_time_idx, note_idx, notes_velocity))

        return events

    all_events = []

    if right_hand_track is not None:
        right_events = extract_events_from_track(right_hand_track, 1, staff_num)  # 1 for right
        all_events += right_events

    all_events.sort(key=lambda x: (x[0], x[2]))  # sort by time, then velocity

    if len(all_events) == 1:
        all_events = []

    if not all_events:
        raise Exception("No events found for the selected MIDI files.")
        # return []

    sequence = []
    prev_time = 0

    for time, note, velocity in all_events:
        delta_time = time - prev_time
        sequence.append((note, velocity, delta_time))
        prev_time = time

    return sequence
