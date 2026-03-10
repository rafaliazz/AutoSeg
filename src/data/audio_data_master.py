"""
Class to manage one audio track 
"""
from pydub import AudioSegment, silence

class AudioTrackObject:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.silent_segments, self.non_silent_segments = self.get_silent_non_silent_segs()
    
    def get_silent_non_silent_segs(self):
        myaudio = AudioSegment.from_wav(self.audio_path) 

        dBFS = myaudio.dBFS
        silent_segments = silence.detect_silence(myaudio, min_silence_len=300, silence_thresh=dBFS-16)

        silent_segments = [((start/1000),(stop/1000)) for start,stop in silent_segments]

        non_silent_segments = []

        for i in range(len(silent_segments) - 1):
            end_current = silent_segments[i][1]
            start_next = silent_segments[i+1][0]
            non_silent_segments.append((end_current, start_next))

        return silent_segments, non_silent_segments