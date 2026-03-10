"""
Class to run the main pipeline used within the API. 
It returns the raw transcription -> segmentation results -> confidence-based resegmentation  
"""
from data.audio_data_master import AudioTrackObject 
from models.fw_model import FasterWhisper
class MainPipeline: 
    def __init__(self, audio_path, model_config=None, confi_thres = -0.25, resegment_minim = 2):
        self.audio_path = audio_path
        self.confi_thres = confi_thres
        self.resegment_minim = resegment_minim
        if model_config is None:
            model_config = {
                "model_size": "small",
                "device": "cpu",
                "compute_type": "int8"
                }

        self.model = FasterWhisper(**model_config)
        self.audio_track_object = AudioTrackObject(audio_path)
        self.raw_transcriptions = self.run_raw_transcription()
        self.silent_segs, self.non_silent_segs = self.segment()
        self.speech_transcriptions = self.silent_based_transcription()
        self.final_transcriptions = self.confidence_based_segmentation()

    def return_everything(self): 
        return self.raw_transcriptions, self.silent_segs, self.non_silent_segs, self.speech_transcriptions, self.final_transcriptions
    
    def run_raw_transcription(self):
        return self.model.transcribe_full(self.audio_path)
    
    def segment(self):
        silent_segs, non_silent_segs = self.audio_track_object.get_silent_non_silent_segs()
        return silent_segs, non_silent_segs

    def silent_based_transcription(self):
        new_transcriptions = []

        for start, end in self.non_silent_segs:
            if end - start < 0.3: 
                continue 
            segments = self.model.transcribe_segment(
                self.audio_path,
                start,
                end
            )

            for seg in segments:
                new_transcriptions.append({
                    "start": seg["start"] + start,
                    "end": seg["end"] + start,
                    "text": seg["text"],
                    "confidence": seg["confidence"]
                })

        return new_transcriptions
    
    def confidence_based_segmentation(self):
        resemented_transcriptions = []

        for segment in self.speech_transcriptions:

            duration = segment["end"] - segment["start"]

            if segment["confidence"] < self.confi_thres and duration >= self.resegment_minim:

                new_segments = self.model.transcribe_segment(
                    self.audio_path,
                    segment["start"],
                    segment["end"]
                )

                for subseg in new_segments:
                    subseg["start"] += segment["start"]
                    subseg["end"] += segment["start"]

                    resemented_transcriptions.append(subseg)

            else:
                resemented_transcriptions.append(segment)

        return resemented_transcriptions
if __name__ == "__main__":
    """Test main"""
    audio_path = "dataset/sample.wav"

    pipeline = MainPipeline(audio_path)

    raw_transcriptions, silent_segs, non_silent_segs, speech_transcriptions, new_transcriptions = pipeline.return_everything()

    print("\n=== RAW TRANSCRIPTIONS ===")
    for seg in raw_transcriptions:
        print(seg)

    print("\n=== SILENT SEGMENTS ===")
    print(silent_segs)

    print("\n=== NON-SILENT SEGMENTS ===")
    print(non_silent_segs)

    print("\n=== SPEECH TRANSCRIPTIONS ===")
    print(speech_transcriptions)

    print("\n=== CONFIDENCE RESEGMENTED TRANSCRIPTIONS ===")
    for seg in new_transcriptions:
        print(seg)

