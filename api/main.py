import random
import torch
from config import *
# from src.data_reader import JSONfiles
# from src.model import Neural_Net
from src.utils import bag_of_words, tokenize

from typing import Union
from fastapi import FastAPI

# uvicorn main:app --reload

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World!"}

@app.get("/predict/{human_sentence}")
def predict(human_sentence: str):
    human_sentence = tokenize(human_sentence)
    X = bag_of_words(human_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents_data['intents']:
            if tag == intent["tag"]:
                bot_answer = random.choice(intent['responses'])
    else:
        bot_answer = "I do not understand..."

    return {"bot_answer": bot_answer}
