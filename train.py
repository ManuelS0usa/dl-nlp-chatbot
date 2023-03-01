import torch
import mlflow
from torch.utils.data import DataLoader
from config import *
from src.data_reader import JSONfiles
from src.data_processing import Training_Data
from src.model import Train_Neural_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set mlflow model training registry
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.get_tracking_uri()
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))
# https://mohameddhaoui.github.io/dataengineering/mlflow/

def train_pipeline():    
    """ Training pipeline """
    
    # Ingest data
    file_path = DATA_FOLDER + '/' + FILE_NAME
    intents_data = JSONfiles(file_path).read()

    # Build training dataset
    training_data = Training_Data(intents_data)
    all_words, tags, xy = training_data.build()  # nlp data transform to batch training

    # Pytorch model and training
    train_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    input_size = training_data.get_input_size()
    output_size = len(tags)
    neural_net_shape = [input_size, HIDDEN_SIZE, output_size]
    Train_Neural_Net(all_words, tags).run(device, train_loader, neural_net_shape)
    
if __name__ == "__main__":
    train_pipeline()