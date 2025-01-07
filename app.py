from fastapi import FastAPI, File, UploadFile, Depends
from transformers import WhisperProcessor,WhisperForConditionalGeneration
import torch
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
import os
import uvicorn
import librosa

ml_models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on ",device)
    ml_models['processor'] = WhisperProcessor.from_pretrained("output/whisper-large-v3-ar_v1")
    ml_models['model'] = WhisperForConditionalGeneration.from_pretrained(
        "output/whisper-large-v3-ar_v1/checkpoint-39000", )
    ml_models['model'].to(device)
    ml_models['device'] = device
    print("Model and processor loaded successfully!")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

def split_audio(audio_path, chunk_duration=30):
    audio, sr = librosa.load(audio_path, sr=16000)
    chunks = []
    for i in range(0, len(audio), chunk_duration * sr):
        chunks.append(audio[i:i + chunk_duration * sr])
    return chunks
    
@app.post("/v1/transcribe")
async def recognize_audio(file: UploadFile = File(...)):
    """
    Endpoint to perform voice recognition on an uploaded audio file.
    """
    processor = ml_models['processor']
    model = ml_models['model']
    device = ml_models['device']
    try:
        # Save the uploaded file to a temporary file
        temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_file.name, "wb") as f:
            f.write(file.file.read())

        # Load the audio file and preprocess
        chunks = split_audio(temp_file.name, chunk_duration=30)
        # speech_array, sampling_rate = librosa.load(temp_file.name, sr=16000)
        # speech_arrSay, sampling_rate = processor.audio_preprocessor()
        
        # Tokenize input
        inputs = processor(chunks, sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform transcription
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Remove the temporary file
        os.remove(temp_file.name)

        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
