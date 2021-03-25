# yelp-sentiment-analysis
PyTorch implementation of LSTM and fine-tuned BERT, XLNet sequence classification models

This is a learning class project for "Deep Learning for NLP with PyTorch". Code is heavily inspired from various notebooks at https://github.com/ravi-ilango/aicamp-mar-2021

The LSTM model achieves approx 88% accuracy on the test set, while the BERT and XLNet-based models achieve 95% and 96% accuracy, respectively.

Dataset: https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz

# Remarks:

1) Training was done on a 3090 GPU (24GB of memory). Despite this, the BERT and XLNet model fine-training necessitates relatively small batch sizes (approx 20). With the LSTM, I was able to push much larger batches (batch size of 2000 was successful).

2) For the two transformer models, since I was only fine-tuning, I only ran training for 3 epochs. Each epoch took between 40 mins (BERT:38 minutes) to an hour (XLNET: 55 mins).

3) As the LSTM model has far fewer paramters, each epoch was fast (less than 30 seconds). However, the accuracy plateaued pretty fast. One config was ran for a 100 epochs and that did not produce noticibly better results (despite adding dropout).

4) As the dataset is approx half a million reviews, running the BERT/XLNet tokenizer at the outset on the entire dataset was sub-optimal. Instead, I implemented a custom Dataset class and used the Dataloader class's multiple worker feature to do tokenization on-demand. This can be improved further via memoization but seemed to be a pre-mature optimization.

5) Since this was a short learning project, I have intentionaly ignored basic code hygine. #DONTJUDGEMEGITHUB
