from music21 import converter, tempo, instrument, stream



if __name__ == '__main__':
    for i in range(1, 9):
        input_xml = f"src/all_data/model_generated/audiveris/musicxml/hi_res/song-{i}.mxl"
        output_midi = f"src/all_data/model_generated/audiveris/hi_res/song-{i}.mid"

        score = converter.parse(input_xml)

        score.insert(0, tempo.MetronomeMark(number=120))

        for part in score.parts:
            part.insert(0, instrument.Piano())

        score.write("midi", output_midi)

        print("Wygenerowano MIDI z pianinem i tempem 120 BPM")
