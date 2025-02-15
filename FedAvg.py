# Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
learning_rate = 0.01
momentum = 0.9
num_epochs = 10
batch_size = 32
number_of_samples = 5
amount_per_label = 100 

# NN architecture 
class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(784, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train a model for one epoch
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

# Function to validate the model
def validate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy

# Function to split and shuffle labels 
def split_and_shuffle_labels(y_data, seed, amount):
    y_data=pd.DataFrame(y_data, columns=["labels"])
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()

    np.random.seed(7)  

    # Count available samples for each label
    label_counts = y_data['labels'].value_counts()

    for i in range(10):  # Labels are from 0 to 9
        var_name="label" + str(i)

        if i not in label_counts:
            print(f"Warning: No samples available for label {i}. Skipping.")
            continue

        available_samples = min(amount, label_counts[i])  # Limit to available samples

        label_info=y_data[y_data["labels"]==i]

        label_info=np.random.permutation(label_info.values)[:available_samples]
        label_info=pd.DataFrame(label_info, columns=["labels","i"])
        label_dict.update({var_name: label_info})

    return label_dict

# Function to get IID subsamples indices 
def get_iid_subsamples_indices(label_dict, number_of_samples):
    sample_dict= dict()

    batch_size=int(np.floor(amount_per_label/number_of_samples))

    for i in range(number_of_samples):
        sample_name="sample"+str(i)
        dumb=pd.DataFrame()

        for j in range(10):  # Labels are from 0 to 9
            label_name=str("label")+str(j)
            if label_name in label_dict:
                a=label_dict[label_name][i*batch_size:(i+1)*batch_size]
                dumb=pd.concat([dumb,a], axis=0)

        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})

    return sample_dict

# Function to create IID subsamples 
def create_iid_subsamples(sample_dict, x_data, y_data):
    x_data_dict= dict()
    y_data_dict= dict()

    for i in range(len(sample_dict)):
        sample_name="sample"+str(i)

        indices=np.sort(np.array(sample_dict[sample_name]["i"]))

        x_info= x_data[indices,:]
        x_data_dict.update({sample_name : torch.tensor(x_info)})

        y_info= y_data[indices]
        y_data_dict.update({sample_name : torch.tensor(y_info)})

    return x_data_dict, y_data_dict

# Function to create model optimizer criterion dict 
def create_model_optimizer_criterion_dict(number_of_samples):
    model_dict = dict()
    optimizer_dict= dict()

    for i in range(number_of_samples):
        model_name="model"+str(i)
        model_info=Net2nn()
        model_dict.update({model_name : model_info })

        optimizer_name="optimizer"+str(i)
        optimizer_info = optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name : optimizer_info })

    return model_dict, optimizer_dict

# Function to get averaged weights 
def get_averaged_weights(model_dict, number_of_samples):
    fc1_mean_weight = torch.zeros_like(model_dict["model0"].fc1.weight.data)
    fc1_mean_bias = torch.zeros_like(model_dict["model0"].fc1.bias.data)

    fc2_mean_weight = torch.zeros_like(model_dict["model0"].fc2.weight.data)
    fc2_mean_bias = torch.zeros_like(model_dict["model0"].fc2.bias.data)

    fc3_mean_weight = torch.zeros_like(model_dict["model0"].fc3.weight.data)
    fc3_mean_bias = torch.zeros_like(model_dict["model0"].fc3.bias.data)

    with torch.no_grad():
        for i in range(number_of_samples):
            fc1_mean_weight += model_dict[f"model{i}"].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[f"model{i}"].fc1.bias.data.clone()

            fc2_mean_weight += model_dict[f"model{i}"].fc2.weight.data.clone()
            fc2_mean_bias += model_dict[f"model{i}"].fc2.bias.data.clone()

            fc3_mean_weight += model_dict[f"model{i}"].fc3.weight.data.clone()
            fc3_mean_bias += model_dict[f"model{i}"].fc3.bias.data.clone()

        fc1_mean_weight /= number_of_samples
        fc1_mean_bias /= number_of_samples

        fc2_mean_weight /= number_of_samples
        fc2_mean_bias /= number_of_samples

        fc3_mean_weight /= number_of_samples
        fc3_mean_bias /= number_of_samples

    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias

# Function to set averaged weights as main model weights
def set_averaged_weights_as_main_model_weights(main_model, model_dict):
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = get_averaged_weights(model_dict, number_of_samples)

    with torch.no_grad():
        main_model.fc1.weight.data.copy_(fc1_mean_weight.data.clone())
        main_model.fc2.weight.data.copy_(fc2_mean_weight.data.clone())
        main_model.fc3.weight.data.copy_(fc3_mean_weight.data.clone())

        main_model.fc1.bias.data.copy_(fc1_mean_bias.data.clone())
        main_model.fc2.bias.data.copy_(fc2_mean_bias.data.clone())
        main_model.fc3.bias.data.copy_(fc3_mean_bias.data.clone())

# Initialize datasets 
x_train_data = np.random.rand(1000, 784).astype(np.float32)
y_train_data = np.random.randint(0, 10, size=(1000)).astype(np.int64)

# Split and shuffle labels to create IID samples
label_dict = split_and_shuffle_labels(y_train_data, seed=42,
                                      amount=amount_per_label * number_of_samples)
sample_indices = get_iid_subsamples_indices(label_dict,
                                             number_of_samples=number_of_samples)

# Create IID samples from the indices generated above
x_train_iid_samples, y_train_iid_samples = create_iid_subsamples(sample_indices,
                                                                  x_train_data,
                                                                  y_train_data)

# Create models and optimizers for each local dataset
model_dict, optimizer_dict = create_model_optimizer_criterion_dict(number_of_samples)

# Training loop for federated learning using FedAvg
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train each local model on its dataset
    for i in range(number_of_samples):

        if f'sample{i}' not in x_train_iid_samples or f'sample{i}' not in y_train_iid_samples:
            print(f"Warning: Sample {i} has no data.")
            continue

        train_ds = TensorDataset(x_train_iid_samples[f'sample{i}'], y_train_iid_samples[f'sample{i}'])

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        train_loss, train_accuracy = train(model_dict[f'model{i}'], train_dl,
                                           nn.CrossEntropyLoss(), optimizer_dict[f'optimizer{i}'])

        print(f"Model {i + 1} | Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")

# Validate the final centralized model (optional)
test_accuracy_list = []
for i in range(number_of_samples):

    x_test_data = np.random.rand(200, 784).astype(np.float32)
    y_test_data = np.random.randint(0, 10, size=(200)).astype(np.int64)

    test_ds = TensorDataset(torch.tensor(x_test_data), torch.tensor(y_test_data))

    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    test_loss, test_accuracy = validate(model_dict[f'model{i}'], test_dl,
                                        nn.CrossEntropyLoss())

    print(f"Model {i + 1} Test Accuracy: {test_accuracy:.4f}")

test_accuracy_list.append(test_accuracy)

print(f"Final Test Accuracy across all models: {np.mean(test_accuracy_list):.4f}")
