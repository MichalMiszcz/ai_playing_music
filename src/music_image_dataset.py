import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import mido

class MusicImageDataset(Dataset):
    def __init__(self, image_root, midi_root, image_transform=None, max_seq_len=100, max_midi_files=100):
        self.image_root = image_root
        self.midi_root = midi_root
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
                midi_seq = extract_notes_from_midi(midi_file)
                if len(midi_seq) > self.max_seq_len:
                    midi_seq = midi_seq[:self.max_seq_len]
                else:
                    midi_seq += [(0, 0)] * (self.max_seq_len - len(midi_seq))
                self.midi_features[midi_key] = midi_seq
            except Exception as e:
                print(f"Error processing MIDI {midi_file}: {e}")
                records_to_remove.append((folder, author, midi_file))
                continue

            file = os.path.splitext(os.path.basename(midi_file))[0]
            image_dir = os.path.join(image_root, author, file)
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
                for file in image_files:
                    self.image_paths.append(os.path.join(image_dir, file))

        for record in records_to_remove:
            self.selected_midi_files.remove(record)

        all_durations = [duration for midi_seq in self.midi_features.values() for _, duration in midi_seq if duration > 0]
        self.max_duration = max(all_durations) if all_durations else 1

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

        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        midi_seq = self.midi_features.get(midi_key, [(0, 0)] * self.max_seq_len)

        normalized_seq = [(note / 127.0, duration / self.max_duration) for note, duration in midi_seq]
        midi_tensor = torch.tensor(normalized_seq, dtype=torch.float32)

        return image, midi_tensor


def extract_notes_from_midi(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None

    right_hand_track = None
    left_hand_track = None

    for track in mid.tracks:
        if track.name == "Piano right":
            right_hand_track = track
        elif track.name == "Piano left":
            left_hand_track = track

    if right_hand_track is None or left_hand_track is None:
        print(f"MIDI file {midi_path} does not have both 'Piano right' and 'Piano left' tracks.")
        return None

    def extract_notes_from_track(track):
        notes = []
        current_time = 0
        note_starts = {}
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_starts[msg.note] = current_time
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                if msg.note in note_starts:
                    start_time = note_starts.pop(msg.note)
                    duration = current_time - start_time
                    notes.append((start_time, msg.note, duration))
        return notes

    right_hand_notes = extract_notes_from_track(right_hand_track)
    left_hand_notes = extract_notes_from_track(left_hand_track)

    all_notes = right_hand_notes + left_hand_notes
    all_notes.sort(key=lambda x: x[0])

    notes = [(note, duration) for _, note, duration in all_notes]

    return notes