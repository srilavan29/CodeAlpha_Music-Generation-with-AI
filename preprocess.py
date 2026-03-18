import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

def get_notes():
    """ Get all the notes and chords from the midi files in the ./data directory """
    notes = []

    for file in sorted(glob.glob("data/*.mid"))[:2]:
        try:
            midi = converter.parse(file)
            print(f"Parsing {file}...")

            # Get all notes and chords in a flat structure
            notes_to_parse = midi.flat.notes

            file_notes = 0
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    file_notes += 1
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    file_notes += 1
            
            print(f"Extracted {file_notes} notes from {file}. Total so far: {len(notes)}")
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    with open('processed_data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with PyTorch (batch, seq_len)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    
    # Save the mapping
    with open('processed_data/mapping.pkl', 'wb') as f:
        pickle.dump(note_to_int, f)

    return network_input, network_output

if __name__ == '__main__':
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    
    np.save('processed_data/network_input.npy', network_input)
    np.save('processed_data/network_output.npy', network_output)
    print("Preprocessing complete. Data saved in processed_data/")
