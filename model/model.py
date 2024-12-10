from pathlib import Path
config_path = Path(__file__).parent / "config.json"
with open(config_path, "r") as file:
    import json
    config = json.load(file)

from dataclasses import dataclass
@dataclass
class ModelPrediction:
    command: str
    option: int
    score: float

from transformers import pipeline
from fastapi import UploadFile
from pydub import AudioSegment
def load_model():
    model_pl = pipeline(config['task'], model=config['model'])
    def model(data: UploadFile) -> ModelPrediction:
        audio = load_audio_data(data)
        pred = model_pl(audio)
        return ModelPrediction(command=pred['text'])
    return model

import numpy as np
def normalize_audio_data(audio, target_db=-0.5):
    peak = np.max(np.abs(audio))
    return audio * (10 ** (target_db / 20.) / peak)

def load_audio_data(data: UploadFile) -> np.array:
    audio = AudioSegment.from_wav(data.file)
    audio = audio.set_frame_rate(16000)
    audio = np.array(audio.get_array_of_samples())
    return normalize_audio_data(audio)