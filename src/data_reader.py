import json
from config import *


class JSONfiles():
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read(self):
        with open(self.file_path, 'r') as f:
            intents = json.load(f)
        return intents
