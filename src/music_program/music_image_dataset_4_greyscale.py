import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import mido
from src.music_program.global_variables import WHITE_KEYS_MIDI, NUM_NOTES

note_to_index = {midi_num: i for i, midi_num in enumerate(WHITE_KEYS_MIDI)}

class MusicImageDataset(Dataset):
    def __init__(self, image_root, midi_root, left_hand_tracks=["Piano left"], right_hand_tracks=["Piano right"], image_transform=None, max_seq_len=100, max_midi_files=100):
        self.image_root = image_root
        self.midi_root = midi_root
        self.left_hand_tracks = left_hand_tracks
        self.right_hand_tracks = right_hand_tracks
        self.image_transform = image_transform if image_transform else transforms.ToTensor()
        self.max_seq_len = max_seq_len

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
        records_to_remove = []

        for folder, author, midi_file in self.selected_midi_files:
            midi_name = os.path.splitext(os.path.basename(midi_file))[0]
            midi_key = f"{author}/{midi_name}"
            try:
                midi_seq = extract_notes_from_midi(midi_file, self.left_hand_tracks, self.right_hand_tracks)
                if midi_seq is None:
                    records_to_remove.append((folder, author, midi_file))
                    continue
                if len(midi_seq) > self.max_seq_len:
                    midi_seq = midi_seq[:self.max_seq_len]
                else:
                    midi_seq += [(0, 0, 0)] * (self.max_seq_len - len(midi_seq))

                self.midi_features[midi_key] = midi_seq
            except Exception as e:
                print(f"Error processing MIDI {midi_file}: {e}")
                records_to_remove.append((folder, author, midi_file))
                continue

            folder = os.path.splitext(os.path.basename(midi_file))[0]
            file = os.path.splitext(os.path.basename(midi_file))[0] + "-1" # Added only for my files
            image_dir = os.path.join(image_root, author, folder)

            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
                for file in image_files:
                    self.image_paths.append(os.path.join(image_dir, file))

        for record in records_to_remove:
            self.selected_midi_files.remove(record)

        all_delta_times = [delta_time for seq in self.midi_features.values() for _, _, delta_time in seq if delta_time > 0]
        self.max_delta_time = max(all_delta_times) if all_delta_times else 1

        self.image_paths.sort()
        print(f"Selected {len(self.selected_midi_files)} MIDI files and {len(self.image_paths)} images.")

        if len(self.image_paths) == 0:
            raise ValueError("No images found for the selected MIDI files. Check directory paths and file structure.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        rel_path = os.path.relpath(img_path, self.image_root)
        composer, piece, _ = rel_path.split(os.sep)
        midi_key = f"{composer}/{piece}"

        image = Image.open(img_path).convert('L')
        if self.image_transform:
            image = self.image_transform(image)

        midi_seq = self.midi_features.get(midi_key, [(0, 0, 0)] * self.max_seq_len)

        normalized_seq = [
            (note_idx / (NUM_NOTES - 1.0),  # Normalize 0-7 to the range [0, 1]
             velocity / 127.0,
             delta_time / self.max_delta_time)
            for note_idx, velocity, delta_time in midi_seq
        ]
        midi_tensor = torch.tensor(normalized_seq, dtype=torch.float32)

        return image, midi_tensor

def extract_notes_from_midi(midi_path, left_hand_tracks, right_hand_tracks):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None

    left_hand_track = None
    right_hand_track = None

    print(midi_path)

    for track in mid.tracks:
        if track.name in left_hand_tracks and left_hand_track is None:
            left_hand_track = track
        elif track.name in right_hand_tracks and right_hand_track is None:
            right_hand_track = track

    if left_hand_track is None and right_hand_track is None:
        print(f"MIDI file {midi_path} does not have the required tracks.")
        return None

    def extract_events_from_track(track, hand):
        events = []
        current_time = 0
        for msg in track:
            current_time += msg.time

            if msg.type in ('note_on', 'note_off'):
                if msg.note in note_to_index:
                    print(msg)
                    if msg.note in note_to_index:
                        note_idx = note_to_index[msg.note]  # Convert MIDI 60, 62... to values 0, 1...
                        notes_velocity = msg.velocity if msg.type == 'note_on' else 0
                        events.append((current_time, note_idx, notes_velocity))
        return events

    all_events = []

    if right_hand_track is not None:
        right_events = extract_events_from_track(right_hand_track, 1)  # 1 for right
        all_events += right_events

    all_events.sort(key=lambda x: (x[0], x[2]))  # sort by time, then velocity

    if not all_events:
        return []

    sequence = []
    prev_time = 0

    for time, note, velocity in all_events:
        delta_time = time - prev_time
        sequence.append((note, velocity, delta_time))
        prev_time = time

    return sequence