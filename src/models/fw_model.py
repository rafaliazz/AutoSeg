"""
Class to handle logic for the Faster Whisper model (STT model used to transcribe)
"""

from faster_whisper import WhisperModel
import librosa

class FasterWhisper:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        """
        Start the FasterWhisper model once so we don't have to reconfigure it again alter on. 
        Just to save time. 
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    def transcribe_full(self, audio_path):
        """
        Transcribe the entire audio file (raw transcription).
        """

        segments, info = self.model.transcribe(audio_path)

        results = []

        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": getattr(segment, "avg_logprob", None)
            })

        return results

    def transcribe_segment(self, audio_path, start, end):
        """
        Transcribe segmwents of the audio using our model 
        """
        full_audio, sr = librosa.load(audio_path) 
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio = full_audio[start_sample:end_sample]
        segments, info = self.model.transcribe(audio)

        results = []
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": getattr(segment, "avg_logprob", None)
            })
        return results 
    