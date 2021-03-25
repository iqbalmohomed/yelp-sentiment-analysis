# Yelp Reviews Regression Model

# Read in Training and Test datasets (CSV format)
# Compute Validation Split
# Basic Idea. Use 3 different encoders (BERT, XLNET and LSTM) to create encoding for each review.
# To aid comparision, we use a constant architecture after the encoder, which is an MLP layer followed by a binary cross entropy loss computation

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


from tqdm import trange

xlnet_model_path = './xlnet_yelpreviews_state_dictv1.pth'

data_dir = '/home/iqbal/nlpclassproject/yelp_review_polarity_csv'

print (data_dir)

df_test = pd.read_csv(data_dir + "/test.csv",names=["Score", "Text"])

def print_df_stats(df,label=None):
    if label:
        print (label)
    print("Shape", df.shape)
    print ("Columns", df.columns)
    print (df.groupby("Score").count())
    print ("")

print_df_stats(df_test, "Testing Data")

df_test.Text = df_test.Text.str.slice(0,512)

df_test.Score = df_test.Score - 1


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

print("Starting XLNet Tokenization")

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', max_length=512)
dataset = YelpReviewsDataset(tokenizer)
print ("LENGTH",len(dataset))
print ("sample", dataset[1])
print ("sample", dataset[2])

dataset_test = YelpReviewsDataset(tokenizer,test=True)
print ("LENGTH",len(dataset))
print ("sample", dataset[1])
print ("sample", dataset[2])

#padded_sequences = tokenizer(list(review_text_train), padding=True)
#print (f"tokenized inputs {padded_sequences['input_ids'][0]}")

#input_ids = padded_sequences['input_ids']
#attention_masks = padded_sequences['attention_mask']

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

batch_size = 20

# Create an iterator of train data with torch DataLoader
train_sampler = RandomSampler(train_set)
train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size,collate_fn=collate_fn,num_workers=8)

# Create an iterator of validation data with torch DataLoader
validation_sampler = SequentialSampler(val_set)
validation_dataloader = DataLoader(val_set, sampler=validation_sampler, batch_size=batch_size,collate_fn=collate_fn,num_workers=8)

# specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print (torch.cuda.get_device_name(0))

num_labels = 2

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",
                                                      num_labels=num_labels)
model.to(device)

# BERT fine-tuning parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=2e-5)


def train(model, iterator, optimizer):
    model.train()

    epoch_loss = 0

    t0 = time.time()

    for step, batch in enumerate(iterator):

        # retrieve input data
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # resets the gradients after every batch
        optimizer.zero_grad()

        # Forward pass
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = output['loss']

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


# Evaluate
def evaluate(model, iterator):
    # initialize every epoch
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    for batch in iterator:
        # retrieve input data
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # deactivates autograd
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output['logits']

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        epoch_acc += tmp_eval_accuracy

    return epoch_acc / len(iterator)

print("Starting Training Loop")

N_EPOCHS = 3
best_valid_acc = 0

# BERT training loop
for _ in trange(N_EPOCHS, desc="Epoch"):

    # train the model
    train_loss = train(model, train_dataloader, optimizer)

    # evaluate the model
    valid_acc = evaluate(model, validation_dataloader)

    # save the best model
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        print("saving model ...")
        torch.save(model.state_dict(), xlnet_model_path)

    print(f'\t Train Loss: {train_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%')

print("Evaluating on Test Data")

# Check performance on test set
review_text_test = df_test.Text.values
labels_test = df_test.Score.values

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,max_length=512)
padded_sequences_test = tokenizer(list(review_text_test), padding=True)

input_ids_test = padded_sequences_test['input_ids']
attention_masks_test = padded_sequences_test['attention_mask']


# Convert data into torch tensors
test_inputs = torch.tensor(input_ids_test)
test_labels = torch.tensor(labels_test)
test_masks = torch.tensor(attention_masks_test)
# Create an iterator of validation data with torch DataLoader
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


test_acc = evaluate(model, test_dataloader)
print(f'Test Acc: {test_acc * 100:.2f}%')
