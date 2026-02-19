import random

random_notes = random.choices(range(1, 9), k=10)
random_times = random.choices(range(1000, 10000), k=10)

lstm_sequence = [[n, dt] for n in random_notes for dt in random_times]
print(lstm_sequence)

changed_lstm_sequence = []
for note, delta_time in lstm_sequence:
    velocity = 0
    changed_lstm_sequence.append([note, velocity, delta_time])
    velocity = 90
    changed_lstm_sequence.append([note, velocity, delta_time])

print(changed_lstm_sequence)
