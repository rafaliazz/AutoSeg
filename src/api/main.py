"""
Class to run the API Service 
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pathlib import Path
from contextlib import asynccontextmanager

from models.fw_model import FasterWhisper
from pipeline.main_pipeline import MainPipeline


UPLOAD_DIR = Path(r"dataset\\uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads model once 
    """

    app.state.whisper_model = FasterWhisper(
        model_size="small",
        device="cpu",
        compute_type="int8"
    )
    yield

app = FastAPI(
    title="AutoSeg",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "AutoSeg API running"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),

    # Pipeline parameters
    confi_thres: float = -0.2,
    resegment_minim: float = 2,

    # Model configuration
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8"
):
    """
    Upload an audio file, run raw transcription → silence segmentation → confidence-based resegmentation.
    """

    file_path = UPLOAD_DIR / file.filename

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        model_config = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type
        }

        pipeline = MainPipeline(
            audio_path=str(file_path),
            model_config=model_config,
            confi_thres=confi_thres,
            resegment_minim=resegment_minim
        )

        raw, silent, non_silent, speech, final = pipeline.return_everything()

        return {
            "raw_transcriptions": raw,
            "silent_segments": silent,
            "non_silent_segments": non_silent,
            "speech_transcriptions": speech, 
            "confidence_resegmented_transcriptions": final
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))