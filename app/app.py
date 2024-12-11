from fastapi import FastAPI
app = FastAPI()

@app.on_event("startup")
def startup_event():
    global model
    from model import load_model
    model = load_model()

@app.get("/")
def index():
    return {"text": "Arduino audio command analysis"}

from pydantic import BaseModel
class ModelResponse(BaseModel):
    command: int
    option: int

from fastapi import UploadFile
@app.get("/predict")
def predict_sentiment(file: UploadFile):
    pred = model(file)
    return ModelResponse(command=pred.command, option=pred.option)