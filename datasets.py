import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['text_column']  # Replace with actual text column name
        label = self.data.iloc[index]['label_column']  # Replace with actual label column name
        encoding = self.tokenizer.encode_plus(
            text, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return encoding['input_ids'][0], encoding['attention_mask'][0], label.astype('int64')
