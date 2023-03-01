import torch
import torch.nn as nn
from config import *
import mlflow
import mlflow.pytorch


class Neural_Net(nn.Module):
    """ Build artificial neural network architecture """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(Neural_Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


class Train_Neural_Net():
    """ Train neural network """

    def __init__(self, all_words, tags):
        self.all_words = all_words
        self.tags = tags

    def set_mlflow_experiment(self):        
        """ Mlflow set experiment """
        return mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    def loss_criteria(self):
        return nn.CrossEntropyLoss()
        
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def routine(self, model, device, train_loader):
        """ Back propagation training """
        # Loss criteria and optimizer
        criterion = self.loss_criteria()
        optimizer = self.optimizer(model)

        # Train the model
        for epoch in range(N_EPOCHS):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                
                # Forward pass
                outputs = model(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]                
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {loss.item():.4f}')
                
        print(f'final loss: {loss.item():.4f}')

    def run(self, device, train_loader, nn_shape):
        """ Run training experiment """
        # neural net shape
        input_size, hidden_size, output_size = nn_shape[0], nn_shape[1], nn_shape[2]
        # neural net architecture
        model = Neural_Net(nn_shape[0], nn_shape[1], nn_shape[1]).to(device)  

        print("Training start...")        
        # Set mlflow experiment
        experiment = self.set_mlflow_experiment()   
        with mlflow.start_run(run_name=RUN_NAME) as run:
            # training routine
            final_loss = self.routine(model, device, train_loader)
            # params
            mlflow.log_param("input_size", input_size)
            mlflow.log_param("hidden_size", hidden_size)
            mlflow.log_param("output_size", output_size)
            mlflow.log_param("epochs", N_EPOCHS)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("batch_size", BATCH_SIZE)
            # metrics
            mlflow.log_metric("loss", final_loss)
            # model
            mlflow.pytorch.log_state_dict(model.state_dict, MODEL_NAME)
            mlflow.pytorch.log_model(model, MODEL_NAME)
            # missing register tags and all_words
        
        mlflow.end_run()   
        
        # Get Experiment Details
        print("Training complete...")
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("run_id: {}".format(run.info.run_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))  

        return run
        

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }
# FILE = "data.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')

# def model_train(model, device, train_loader):
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     print("Training start...")
#     # Train the model
#     for epoch in range(N_EPOCHS):
#         for (words, labels) in train_loader:
#             words = words.to(device)
#             labels = labels.to(dtype=torch.long).to(device)
            
#             # Forward pass
#             outputs = model(words)
#             # if y would be one-hot, we must apply
#             # labels = torch.max(labels, 1)[1]                
#             loss = criterion(outputs, labels)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#         if (epoch+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {loss.item():.4f}')
            
#     print(f'final loss: {loss.item():.4f}')
#     print("Training complete...")