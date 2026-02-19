import mido


def make_monophonic(track):
    """
    Takes a MidiTrack and enforces monophony (one note at a time).
    It reorders overlapping notes so the 'Note Off' always comes
    before the next 'Note On'.
    """
    new_track = mido.MidiTrack()

    # We copy meta messages (tempo, track name) directly
    # We process note messages to fix the order

    active_note = None
    pending_time = 0

    for msg in track:
        # Pass non-note messages (like MetaMessage or ProgramChange) through
        if not isinstance(msg, mido.Message) or msg.type not in ['note_on', 'note_off']:
            # If we had time waiting from a deleted note-off, add it here
            if pending_time > 0:
                msg.time += pending_time
                pending_time = 0
            new_track.append(msg)
            continue

        # Standardize: mido treats note_on(vel=0) as note_off
        is_note_on = (msg.type == 'note_on' and msg.velocity > 0)

        # Add any accumulated time from skipped messages to the current message
        current_time = msg.time + pending_time
        pending_time = 0

        if is_note_on:
            # === NEW NOTE STARTING ===
            if active_note is not None:
                # SCENARIO: A note is already playing!
                # We must kill the old note *before* starting the new one.
                # The 'time' (wait) belongs to the silence/hold before this moment.

                # 1. Insert Note-Off for the old note using the wait time
                new_track.append(mido.Message('note_on', note=active_note, velocity=0, time=current_time))

                # 2. Start the new note immediately (time=0)
                new_msg = msg.copy()
                new_msg.time = 0
                new_track.append(new_msg)
            else:
                # No overlap, just play normally
                new_msg = msg.copy()
                new_msg.time = current_time
                new_track.append(new_msg)

            # Update tracker
            active_note = msg.note

        else:
            # === NOTE STOPPING ===
            # (This handles 'note_off' or 'note_on velocity=0')

            if active_note == msg.note:
                # This is a valid stop for the currently playing note.
                new_msg = msg.copy()
                new_msg.time = current_time
                new_track.append(new_msg)
                active_note = None
            else:
                # === THE FIX ===
                # This is a "stale" Note-Off.
                # This happens in your files: We already auto-killed this note
                # when the new note started. We don't need this message anymore.

                # HOWEVER: We must save its 'time' value so the rhythm doesn't break.
                pending_time += current_time

    return new_track


# --- Usage Example ---

# 1. Create your overlapping file (simulating your data)
for num in range (9, 26):

    mid = mido.MidiFile(f'src/all_data/model_generated/scan2notes/low_res/song-{num}.mid')
    mid_sorted = mido.MidiFile()

    print("--- Original (Overlapping) ---")
    for j, track in enumerate(mid.tracks):
        for msg in track:
            print(msg)
            clean_track = make_monophonic(track)

    print("\n--- Cleaned (Monophonic) ---")

    for msg in clean_track:
        print(msg)

    # 3. Save if needed
    mid_sorted.tracks.append(clean_track)
    mid_sorted.save(f'src/all_data/model_generated/scan2notes/low_res_sorted/song-{num}.mid')