# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import torchtext.data as data
import torchtext.datasets as datasets
from collections import Counter, defaultdict, OrderedDict

import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/',
                    help='location of the data corpus')
parser.add_argument('--dict_path', type=str, default='../data/dialogue.vocab.pt',
                    help='location of dictionary')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='evaluation batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

############################
# Load data
############################
print ("Loading data...")

PAD_WORD = '<blank>'
eval_batch_size = args.eval_batch_size

src = data.Field(pad_token=PAD_WORD)
trg = data.Field(pad_token=PAD_WORD)

train_data = datasets.TranslationDataset(path=args.data + '/train', exts=('.en', '.de'), fields=(src, trg))
val_data = datasets.TranslationDataset(path=args.data + '/valid', exts=('.en', '.de'), fields=(src, trg))
test_data = datasets.TranslationDataset(path=args.data + '/test', exts=('.en', '.de'), fields=(src, trg))

print ("DONE\n")

############################
# Load vocab
############################

print ("Loading vocab...")

vocab = dict(torch.load(args.dict_path, "text"))
v = vocab['tgt']
v.stoi = defaultdict(lambda: 0, v.stoi)
src.vocab = v; trg.vocab = v
ntokens = len(v.itos)

print "DONE. Vocab size:", ntokens, "\n"

############################
# Build the model
############################

print ("Building model...")

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()

print ("Done\n")

############################
# Training code
############################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(eval_batch_size)

    total_loss = 0
    batch_counter = 0
    samples_num = len(data_source.examples)
    data_iter = data.BucketIterator(dataset=data_source, batch_size=eval_batch_size,
	sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))

    while(eval_batch_size * batch_counter < samples_num):
	batch = next(iter(data_iter))
	source = batch.src
	targets = batch.trg.view(-1)
        output, hidden = model(source, hidden)
        output_flat = output.view(-1, ntokens)
	ss = []; s = 0; sseq = []
	tt = targets.data.cpu().numpy()
	for i in range(0, eval_batch_size):
	    del sseq[:]
	    for j in range(i, len(output_flat), eval_batch_size):
	    	if not tt[j] == 1:
		    sseq.append(tt[j])
	            s += output_flat[j][tt[j]]
	    ss.append(s)
	    s = 0
        total_loss += len(source) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
	batch_counter += 1

    return total_loss[0] / len(data_source)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    hidden = model.init_hidden(args.batch_size)
    train_iter = data.BucketIterator(dataset=train_data, batch_size=args.batch_size, 
	sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)))
    lr = args.lr
    best_val_loss = None

    total_loss = 0
    epoch = 0
    epoch_start_time = time.time()
    batch_counter = 0
    start_time = time.time()
    samples_num = len(train_data.examples)

    while(epoch < args.epochs):
	batch = next(iter(train_iter))
	source = batch.src
	targets = batch.trg.view(-1)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(source, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch_counter % args.log_interval == 0 and batch_counter > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_counter, len(train_data) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

	# Manully update epoch & batch states
	if args.batch_size * batch_counter > samples_num:
	    epoch += 1
	    val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
	    epoch_start_time = time.time()
	    batch_counter = 0
	batch_counter += 1


# At any point you can hit Ctrl + C to break out of training early.
try:
    train()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
