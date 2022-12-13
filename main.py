# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:27:38.777535Z","iopub.execute_input":"2022-12-13T11:27:38.777910Z","iopub.status.idle":"2022-12-13T11:27:38.785607Z","shell.execute_reply.started":"2022-12-13T11:27:38.777877Z","shell.execute_reply":"2022-12-13T11:27:38.784662Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np
import pandas as pd
import os
import tqdm

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, pad_sequences
import time
import datetime

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 500)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:21:08.240939Z","iopub.execute_input":"2022-12-13T11:21:08.241364Z","iopub.status.idle":"2022-12-13T11:21:09.112704Z","shell.execute_reply.started":"2022-12-13T11:21:08.241327Z","shell.execute_reply":"2022-12-13T11:21:09.110693Z"},"jupyter":{"outputs_hidden":false}}
traindf = pd.read_csv('train.csv')
traindf

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:21:23.776592Z","iopub.execute_input":"2022-12-13T11:21:23.776994Z","iopub.status.idle":"2022-12-13T11:21:23.814804Z","shell.execute_reply.started":"2022-12-13T11:21:23.776958Z","shell.execute_reply":"2022-12-13T11:21:23.813213Z"},"jupyter":{"outputs_hidden":false}}
categories = ['Claim', 'Concluding Statement', 'Counterclaim', 'Evidence', 'Lead', 'Position', 'Rebuttal']
traindf['labels'] = traindf.discourse_type.astype('category').cat.codes
traindf

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:21:45.491808Z","iopub.execute_input":"2022-12-13T11:21:45.492224Z","iopub.status.idle":"2022-12-13T11:21:45.497734Z","shell.execute_reply.started":"2022-12-13T11:21:45.492182Z","shell.execute_reply":"2022-12-13T11:21:45.496166Z"},"jupyter":{"outputs_hidden":false}}
sentences = traindf['discourse_text'].values
labels = traindf['labels'].values

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:21:49.779570Z","iopub.execute_input":"2022-12-13T11:21:49.779967Z","iopub.status.idle":"2022-12-13T11:21:52.145624Z","shell.execute_reply.started":"2022-12-13T11:21:49.779934Z","shell.execute_reply":"2022-12-13T11:21:52.144086Z"},"jupyter":{"outputs_hidden":false}}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Print the original sentence.
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:02:03.974512Z","iopub.execute_input":"2022-12-13T11:02:03.974909Z","iopub.status.idle":"2022-12-13T11:05:25.133071Z","shell.execute_reply.started":"2022-12-13T11:02:03.974877Z","shell.execute_reply":"2022-12-13T11:05:25.131424Z"},"jupyter":{"outputs_hidden":false}}
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for sent in tqdm.tqdm(sentences):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

        # This function also supports truncation and conversion
        # to pytorch tensors, but we need to do padding, so we
        # can't use these features :( .
        max_length=512,  # Truncate all sentences.
        truncation=True
        # return_tensors = 'pt',     # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:05:32.295558Z","iopub.execute_input":"2022-12-13T11:05:32.295930Z","iopub.status.idle":"2022-12-13T11:05:32.329274Z","shell.execute_reply.started":"2022-12-13T11:05:32.295901Z","shell.execute_reply":"2022-12-13T11:05:32.328100Z"},"jupyter":{"outputs_hidden":false}}
print('Max sentence length: ', max([len(sen) for sen in input_ids]))

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:06:54.808915Z","iopub.execute_input":"2022-12-13T11:06:54.809277Z","iopub.status.idle":"2022-12-13T11:06:55.612158Z","shell.execute_reply.started":"2022-12-13T11:06:54.809248Z","shell.execute_reply":"2022-12-13T11:06:55.611066Z"},"jupyter":{"outputs_hidden":false}}
# Set the maximum sequence length.
# I've chosen 64 somewhat arbitrarily. It's slightly larger than the
# maximum training sentence length of 47...
MAX_LEN = 512

print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

# Pad our input tokens with value 0.
# "post" indicates that we want to pad and truncate at the end of the sequence,
# as opposed to the beginning.
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

print('\nDone.')

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:07:39.374954Z","iopub.execute_input":"2022-12-13T11:07:39.375330Z","iopub.status.idle":"2022-12-13T11:08:08.165558Z","shell.execute_reply.started":"2022-12-13T11:07:39.375297Z","shell.execute_reply":"2022-12-13T11:08:08.164128Z"},"jupyter":{"outputs_hidden":false}}
# Create attention masks
attention_masks = []

# For each sentence...
for sent in tqdm.tqdm(input_ids):
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]

    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:22:29.615097Z","iopub.execute_input":"2022-12-13T11:22:29.615586Z","iopub.status.idle":"2022-12-13T11:22:29.866193Z","shell.execute_reply.started":"2022-12-13T11:22:29.615547Z","shell.execute_reply":"2022-12-13T11:22:29.864735Z"},"jupyter":{"outputs_hidden":false}}
# Use train_test_split to split our data into train and validation sets for
# training

# Use 90% for training and 10% for validation.
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                    random_state=2018, test_size=0.1)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                       random_state=2018, test_size=0.1)

train_labels = to_categorical(list(train_labels))
train_labels.shape

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:22:35.552644Z","iopub.execute_input":"2022-12-13T11:22:35.553041Z","iopub.status.idle":"2022-12-13T11:22:39.704121Z","shell.execute_reply.started":"2022-12-13T11:22:35.553007Z","shell.execute_reply":"2022-12-13T11:22:39.703070Z"},"jupyter":{"outputs_hidden":false}}
# Convert all inputs and labels into torch tensors, the required datatype
# for our model.
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:22:49.587347Z","iopub.execute_input":"2022-12-13T11:22:49.587790Z","iopub.status.idle":"2022-12-13T11:22:49.596389Z","shell.execute_reply.started":"2022-12-13T11:22:49.587757Z","shell.execute_reply":"2022-12-13T11:22:49.594365Z"},"jupyter":{"outputs_hidden":false}}
#  The DataLoader needs to know our batch size for training, so we specify it
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:22:54.834840Z","iopub.execute_input":"2022-12-13T11:22:54.835266Z","iopub.status.idle":"2022-12-13T11:23:12.764773Z","shell.execute_reply.started":"2022-12-13T11:22:54.835211Z","shell.execute_reply":"2022-12-13T11:23:12.762337Z"},"jupyter":{"outputs_hidden":false}}
#  Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=7,  # The number of output labels--7for Multi-class classification.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

model

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:24:00.543919Z","iopub.execute_input":"2022-12-13T11:24:00.544343Z","iopub.status.idle":"2022-12-13T11:24:00.554511Z","shell.execute_reply.started":"2022-12-13T11:24:00.544310Z","shell.execute_reply":"2022-12-13T11:24:00.552768Z"},"scrolled":true,"jupyter":{"outputs_hidden":false}}
model

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:25:32.449744Z","iopub.execute_input":"2022-12-13T11:25:32.450105Z","iopub.status.idle":"2022-12-13T11:25:32.459063Z","shell.execute_reply.started":"2022-12-13T11:25:32.450075Z","shell.execute_reply":"2022-12-13T11:25:32.457571Z"},"jupyter":{"outputs_hidden":false}}
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:25:42.599638Z","iopub.execute_input":"2022-12-13T11:25:42.600035Z","iopub.status.idle":"2022-12-13T11:25:42.612233Z","shell.execute_reply.started":"2022-12-13T11:25:42.600002Z","shell.execute_reply":"2022-12-13T11:25:42.610732Z"},"jupyter":{"outputs_hidden":false}}
#  Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:28:54.792045Z","iopub.execute_input":"2022-12-13T11:28:54.792418Z","iopub.status.idle":"2022-12-13T11:28:54.801180Z","shell.execute_reply.started":"2022-12-13T11:28:54.792390Z","shell.execute_reply":"2022-12-13T11:28:54.800249Z"},"jupyter":{"outputs_hidden":false}}
# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


# %% [code] {"jupyter":{"outputs_hidden":false}}
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# %% [code] {"jupyter":{"outputs_hidden":false}}
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# %% [code] {"execution":{"iopub.status.busy":"2022-12-13T11:34:14.160841Z","iopub.execute_input":"2022-12-13T11:34:14.161246Z"},"jupyter":{"outputs_hidden":false}}
import random

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in tqdm.tqdm(enumerate(train_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.numpy()
        label_ids = b_labels.numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")