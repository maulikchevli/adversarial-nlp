# main.py

import torch
import torch.nn as nn
import torchtext as tt
import torch.optim as optim

from torch.utils.model_zoo import tqdm
import copy
import random
import time

from utils import *
from network import RNN

SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'cpu')


def train_batch(model, batch, optimizer, criterion):
    optimizer.zero_grad()
    text, text_lengths = batch.text
    
    y_pred = model(text, text_lengths).squeeze(1)
    loss = criterion(y_pred, batch.label)
    acc = accuracy(y_pred, batch.label)

    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()

def evaluate_batch(model, batch, criterion):
    with torch.no_grad():
        text, text_lengths = batch.text
    
        y_pred = model(text, text_lengths).squeeze(1)
        loss = criterion(y_pred, batch.label)
        acc = accuracy(y_pred, batch.label)

    return loss.item(), acc.item()

def time_min_sec(s, e):
    diff = e - s
    diff_min = int(diff / 60)
    diff_sec = int(diff - (diff_min * 60))

    return diff_min, diff_sec

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def main():
    # Include length for packed padded sequence
    c_text = tt.data.Field(tokenize='spacy', include_lengths=True)
    c_label = tt.data.LabelField(dtype=torch.float)
    
    train_data, test_data = tt.datasets.IMDB.splits(c_text, c_label)
    train_data, val_data = train_data.split(random_state=random.seed(SEED))
    
    VOCAB_SIZE = 15000
    
    # Vocab is lookup table for every word
    c_text.build_vocab(train_data,
                    max_size = VOCAB_SIZE,
                    vectors = 'glove.6B.100d',
                    unk_init = torch.Tensor.normal_)
    c_label.build_vocab(train_data)
    
    batch_size = 64
    train_iterator, \
    val_iterator, \
    test_iterator = tt.data.BucketIterator.splits(
                        (train_data, val_data, test_data),
                        batch_size = batch_size,
                        sort_within_batch = True,
                        device = device)
    
    # Initialize model
    input_dim = len(c_text.vocab)
    embedding_dim = 100 # Should be same as dim of pre trained embeddings
    hidden_dim = 256
    output_dim = 1
    num_layers = 2
    is_bidirectional = True
    dropout_rate = 0.5
    pad_idx = c_text.vocab.stoi[c_text.pad_token]
    
    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim,
                num_layers, is_bidirectional, dropout_rate, pad_idx)

    # initialize embeddings
    pretrained_embeddings = c_text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    unk_idx = c_text.vocab.stoi[c_text.unk_token]
    model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
    print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    criterion.to(device)

    # Train/Validate loop
    EPOCHS = 2
    for epoch in range(EPOCHS):
        # Train
        loss, acc = 0, 0
        model.train()
        pbar = tqdm(total=len(train_iterator) + len(val_iterator))
        for batch in train_iterator:
            batch_loss, batch_acc= train_batch(model, batch, optimizer, criterion)
            loss += batch_loss
            acc += batch_acc
            pbar.update(1)
        
        # Validate
        model.eval()
        val_loss, val_acc = 0, 0
        for batch in val_iterator:
            batch_loss, batch_acc= evaluate_batch(model, batch, criterion)
            val_loss += batch_loss
            val_acc += batch_acc
            pbar.update(1)

        loss, acc = loss/len(train_iterator), acc/len(train_iterator)
        val_loss, val_acc = val_loss/len(val_iterator), val_acc/len(val_iterator)

        pbar.set_postfix_str("e: {}/{} => (T/V) Loss: {:.3f} | {:.3f}, Acc: {:.3f} | {:.3f}\n". format(
            epoch, EPOCHS, loss, val_loss, acc*100, val_acc*100))        

    '''
    Triggers
    '''
    # Attach hooks for gradients
    global extracted_grads
    extracted_grads = []
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.register_backward_hook(extract_grad_hook)

    # Filter target dataset
    dataset_label_filter = "pos"
    targeted_dev_data = []
    for instance in train_data:
        if instance.label == dataset_label_filter:
            targeted_dev_data.append(instance)
    
    target_data = tt.data.Dataset(targeted_dev_data, fields=[('text', c_text), ('label', c_label)])
    target_iter = tt.data.Iterator(target_data, batch_size=batch_size, shuffle = True,
                                    sort_key=lambda x: x.text[1])

    # Trigger tokens
    num_trigger_tokens = 3
    trigger_token_ids = [c_text.vocab.stoi["nice"]] * num_trigger_tokens

    TRIGGER_EPOCHS = 1
    for epoch in range(TRIGGER_EPOCHS):
        # Get Accuracy
        print_string = ""
        for idx in trigger_token_ids:
            print_string = print_string + c_text.vocab.itos[idx] + ', '
        print("Current Triggers: " + print_string + " : " + str(get_accuracy(model, target_iter, criterion,trigger_token_ids, device).item()))
    
        # Get Average grad
        ###
        for batch in target_iter:
            text, text_lengths = batch.text

            optimizer.zero_grad()
            # Reset extracted_grads as in next step we get element at index 0
    
            # RNN needs to be in train() to calculate gradient
            # Hence, disable dropouts manually
            model.train()
            for name, module in model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0
        
                elif isinstance(module, nn.LSTM):
                    module.dropout = 0
        
                elif isinstance(module, nn.GRU):
                    module.dropout = 0
    
            # Append trigger tokens
            trigger_sequence_tensor = torch.LongTensor(copy.deepcopy(trigger_token_ids)).unsqueeze(1) # or 2
            trigger_sequence_tensor = trigger_sequence_tensor.repeat(1, batch.label.shape[0]) # batch_size is global
            b_text = torch.cat((trigger_sequence_tensor, text))
            # Add text length
            b_text_lengths = text_lengths + num_trigger_tokens
    
            y_pred = model(b_text.to(device), b_text_lengths.to(device)).squeeze(1)
            loss = criterion(y_pred, batch.label.to(device))
    
            extracted_grads = []
            loss.backward()
        
            data_grad = extracted_grads[0]
            averaged_grad = torch.sum(data_grad, dim=0)
            # Get gradient of trigger tokens only
            averaged_grad = averaged_grad[0:len(trigger_token_ids)]
    
            # print(averaged_grad.shape)
            cand_trigger_token_ids = hotflip_attack(averaged_grad,
                                                            model.embedding.weight,
                                                            trigger_token_ids,
                                                            num_candidates=2,
                                                            increase_loss=True)
            
            trigger_token_ids = get_best_candidates(model,
                                                        batch,
                                                        criterion,
                                                        trigger_token_ids,
                                                        cand_trigger_token_ids,
                                                        device)
    print_string = ""
    for idx in trigger_token_ids:
        print_string = print_string + c_text.vocab.itos[idx] + ', '
    print("Current Triggers: " + print_string + " : " + str(get_accuracy(model, target_iter, criterion, trigger_token_ids, device).item()))

if __name__ == "__main__":
    main()
