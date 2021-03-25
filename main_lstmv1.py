# Yelp Reviews Sequence Classification Model

# Read in Training and Test datasets (CSV format)
# Compute Validation Split
# Basic Idea: Try 3 different models (BERT, XLNET and LSTM) to do sentiment analysis for each review.

# imports
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
import time
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch.optim as optim


from tqdm import trange

lstm_model_path = './lstm_yelpreviews_state_dictv1.pth'

data_dir = '/home/iqbal/nlpclassproject/yelp_review_polarity_csv'

print (data_dir)

print("Loading CSVs into Dataframes")
df_train = pd.read_csv(data_dir + "/train.csv",names=["Score", "Text"])
df_test = pd.read_csv(data_dir + "/test.csv",names=["Score", "Text"])

df_train['Length'] = df_train['Text'].apply(lambda s: len(s.split()))
df_train.sort_values(by=['Text'], ascending=False, inplace=True)

print("Building Vocab started")
tokenizer = get_tokenizer('basic_english')

counter = Counter()
for _, row in df_train.iterrows():
    counter.update(tokenizer(row['Text']))

vocab = Vocab(counter, min_freq=1)

print("Building Vocab done")
print(f"Size of TEXT vocabulary: {len(vocab)}\n")
print(f"Commonly used words: {vocab.freqs.most_common(10)}\n")


text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x)

class YelpReviewPandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe.Text = self.dataframe.Text.str.slice(0, 512)
        self.dataframe.Score = self.dataframe.Score - 1

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

def generate_batch(batch):
    label_list, text_list, text_length_list = [], [], []
    SEQSIZE = len(batch[0]['Text'].split())
    for row in batch:
        _text = row['Text']
        _label = row['Score']
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)[:SEQSIZE]
        text_length_list.append(len(processed_text))
        processed_text += [1]*(SEQSIZE-len(processed_text))
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.float32)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    text_length_list = torch.tensor(text_length_list, dtype=torch.int64)
    return label_list, text_list, text_length_list

def print_df_stats(df,label=None):
    if label:
        print (label)
    print("Shape", df.shape)
    print ("Columns", df.columns)
    print (df.groupby("Score").count())
    print ("")

#print_df_stats(df_test, "Testing Data")

#df_test.Text = df_test.Text.str.slice(0,512)
#df_test.Score = df_test.Score - 1



class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers,
                 bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embedding = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedding,
                                                            text_lengths.cpu(),
                                                            batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_outputs = self.fc(hidden)
        outputs = self.act(dense_outputs)
        return outputs

def clip(ten,maxlen=512):
    a = ten[:maxlen]
    dif = maxlen - len(a)
    ret = torch.cat((a[:maxlen],torch.zeros(dif,dtype=torch.int64)))
    return ret



class YelpReviewsDataset(Dataset):
    def __init__(self,toklenizer,test=False):
        filename = '/train.csv'
        if test:
            filename = '/test.csv'
        self.df = pd.read_csv(data_dir + filename, names=["Score", "Text"])
        print_df_stats(self.df, "Training Data")
        self.df.Text = self.df.Text.str.slice(0, 512)
        self.df.Score = self.df.Score - 1
        self.review_text_train = self.df.Text.values
        self.labels_train = self.df.Score.values
        self.tokenizer = toklenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        padded_sequence = self.tokenizer(self.review_text_train[idx],padding=True)
        input_ids = padded_sequence['input_ids']
        attention_mask = padded_sequence['attention_mask']
        return torch.tensor(input_ids,dtype=torch.int64),torch.tensor(attention_mask,dtype=torch.int64),torch.tensor(self.labels_train[idx])


dataset = YelpReviewPandasDataset(df_train)
dataset_test = YelpReviewPandasDataset(df_test)
print("Starting Train-Validation Splitting")

train_set, val_set = torch.utils.data.random_split(dataset, [460000, 100000])

def collate_fn(batch):
    label_list, text_list,mask_list = [],[],[]
    maxlen = 0
    for row in batch:
        if len(row[0]) > maxlen:
            maxlen = len(row[0])

    for row in batch:
        label_list.append(row[2])
        text_list.append(clip(row[0],maxlen))
        mask_list.append(clip(row[1],maxlen))

    label_list = torch.tensor(label_list,dtype=torch.int64)
    text_list = torch.stack(text_list)
    mask_list = torch.stack(mask_list)

    return text_list,mask_list,label_list

batch_size = 512

# Create an iterator of train data with torch DataLoader
train_sampler = RandomSampler(train_set)
train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size,collate_fn=generate_batch,num_workers=8)

# Create an iterator of validation data with torch DataLoader
validation_sampler = SequentialSampler(val_set)
validation_dataloader = DataLoader(val_set, sampler=validation_sampler, batch_size=batch_size,collate_fn=generate_batch,num_workers=8)

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print (torch.cuda.get_device_name(0))

num_labels = 2

#define hyperparameters
size_of_vocab = len(vocab)
embedding_dim = 100
hidden_dim = 32
output_dim = 1
num_layers = 2
bidirection = True
dropout = 0.2

#instantiate the model
model = LSTMClassifier(size_of_vocab, embedding_dim, hidden_dim, output_dim, num_layers,
                   bidirectional = True, dropout = dropout)
print("Model Details")
print(model)

model.to(device)

#Check model device type
print(next(model.parameters()).is_cuda, device)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()



def train(model, iterator, optimizer,criterion):
    model.train()

    epoch_loss = 0

    t0 = time.time()

    for step, batch in enumerate(iterator):

        # retrieve input data
        batch = tuple(t.to(device) for t in batch)
        b_labels, b_text, b_text_lengths = batch

        # resets the gradients after every batch
        optimizer.zero_grad()

        # Forward pass
        predictions = model(b_text, b_text_lengths).squeeze()
        loss = criterion(predictions, b_labels)

        # Backward pass
        loss.backward()

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # loss
        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"step: {step}")

    print('{} seconds'.format(time.time() - t0))
    return epoch_loss / len(iterator)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Evaluate
def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_acc = 0
    epoch_loss = 0

    # deactivating dropout layers
    model.eval()

    for batch in iterator:
        # retrieve input data
        batch = tuple(t.to(device) for t in batch)
        b_labels, b_text, b_text_lengths = batch

        # deactivates autograd
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            predictions = model(b_text, b_text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, b_labels)
            acc = binary_accuracy(predictions, b_labels)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)




print("Starting Training Loop")

N_EPOCHS = 100
best_valid_acc = 0

# Model Training loop
for _ in trange(N_EPOCHS, desc="Epoch"):

    # train the model
    train_loss = train(model, train_dataloader, optimizer,criterion)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, validation_dataloader,criterion)

    # save the best model
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        print("saving model ...")
        torch.save(model.state_dict(), lstm_model_path)

    print(f'\t Train Loss: {train_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%')

print("Evaluating on Test Data")

# Check performance on test set
review_text_test = df_test.Text.values
labels_test = df_test.Score.values

# Create an iterator of validation data with torch DataLoader
test_sampler = SequentialSampler(dataset_test)
test_dataloader = DataLoader(dataset_test, sampler=test_sampler, batch_size=batch_size,collate_fn=generate_batch)


test_loss, test_acc = evaluate(model, test_dataloader,criterion)
print(f'Test Acc: {test_acc * 100:.2f}%')
print(f'Test Loss: {test_loss * 100:.2f}%')