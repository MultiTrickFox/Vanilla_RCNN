from music21 import note, chord
from music21 import meter, stream
from music21 import metadata

import interact_midi



def bootstrap():
    extra_prop = input('Song Creation Steps: ')
    extra_prop = 0 if extra_prop == '' else int(extra_prop)
    response = interact_midi.bootstrap(noprint=True, multipropogate=extra_prop)
    if response is not None:
        print_music(response)


def print_music(converted_output):

    thisstream = stream.Stream()
    thisstream.timeSignature = meter.TimeSignature('4/4')
    thisstream.insert(0, metadata.Metadata(
        title='vanilla ai',
        composer='sent from my ai'))

    for timestep in converted_output:

        t1,t2,t3,t4 = timestep

        # hm_elements = len(t1) # todo: also handle rests, as well as optimize for single notes
        thischord = chord.Chord()

        for tt1,tt2,tt3,tt4 in zip(t1,t2,t3,t4):
            tt1.replace('#','+')
                # tt1 = 'R'
            thisnote = note.Note(tt1 + str(tt2+2))
            thisnote.duration.quarterLength = tt3
            thisnote.volume.velocity = tt4
            thischord.add(thisnote)

        thisstream.append(thischord)

    thisstream.show()
    thisstream.plot(title='my musiplot')


if __name__ == '__main__':
    bootstrap()
