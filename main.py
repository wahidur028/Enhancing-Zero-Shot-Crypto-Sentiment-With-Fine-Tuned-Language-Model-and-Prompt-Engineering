import os
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import MyDataset
from model import MyModel
from train import train_model
from test import test_model
from utils import setup_device, set_seed

# Set random seed
set_seed()

# Setup device
device = setup_device()

# Load data
train = pd.read_csv('path_to_train.csv')  # Replace with actual path
test = pd.read_csv('path_to_test.csv')  # Replace with actual path

# Split data
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)

# Hyperparameters
learning_rate = 2e-5
max_length = 512
batch_size = 8
num_labels = 2
epochs = 3
model_ckp = "model_checkpoint"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckp)

# Dataset and loaders
train_dataset = MyDataset(train_data, tokenizer, max_length)
val_dataset = MyDataset(val_data, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model
loss_fn = nn.CrossEntropyLoss()
model = MyModel(model_ckp, num_labels, learning_rate, loss_fn)

# Train
trainer, checkpoint_callback = train_model(model, train_loader, val_loader, epochs)

# Test
test_dataset = MyDataset(test, tokenizer, max_length)
test_model(model, test_dataset, batch_size)

