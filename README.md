# Rhetorical Role Labeling for Legal Judgements

This project was pursued under the guidance of Dr. Yashvardhan Sharma, CSIS Department, BITS Pilani, as a task for submission to the Artificial Intelligence for Legal Assistance (AILA 2020) conference.

> The implementation was carried out in Google Colab notebooks (with a Tesla T4 GPU) for training the models built on the Pytorch framework.

> Pretrained RoBERTa natural language model (a Robust and optimized BERT pretraining approach) was employed for the task of sentence classification.

> The transformers library offered by HuggingFace was also used for building the model.

### Requirements
* torch
* Transformers
* Pandas
* Numpy
* Glob
* time, datetime
* os
### Implementation

The data was loaded from the raw text files into a pandas dataframe with two separate columns for ‘sentence’ and ‘label’. The labels were then enumerated to an integer (0-7) using a python dictionary.
A pretrained RoBERTa Tokenizer(“roberta-base”) was then loaded and applied for each sentence to get the input ids and the corresponding attention masks.
The processed training dataset of the input ids, attention masks and labels was converted into a tensor dataset using the TensorDataset function from the torch.utils.data class, and using the same class a Data loader with a batch size of 16 was created.
The resulting transformer model (a pre-trained base model from the RobertaForSequenceClassification class) was fine tuned over the training dataset to arrive at the optimum parameters. Adam Optimizer was used from the HuggingFace’s AdamW library with the learning rate set to (2e-5) and Adam epsilon set to (1e-8). The seed value was set to 42 and the model was trained for 4 epochs. The norms of the gradients were chipped to 1.0 to help prevent the explosive gradient problem.
Finally, the test data was similarly processed and passed through the model with a batch size of 1.

### License
- MIT License 
Copyright (c) 2020 Team Spectre