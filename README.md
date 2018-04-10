# MMI_anti_pytorch
This project is a pytorch implementation for the MMI-anti model described in [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055v2.pdf)

## Reference <br />
This code is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) and [word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model)

## Quickstart <br />

### Step1: Preprocess the data <br />
```
python preprocess.py
```
We will be working with some example data in `data/` folder. The data consists of parallel dialogue context (`.en`) and its response (`.vi`) data containing one sentence per line with tokens separated by a space:

* `train.en`
* `train.vi`
* `dev.en`
* `dev.vi`

After running the preprocessing, the following files are generated in `data/` folder:

* `dialogue.train.1.pt`: serialized PyTorch file containing training data
* `dialogue.valid.1.pt`: serialized PyTorch file containing validation data
* `dialogue.vocab.pt`: serialized PyTorch file containing vocabulary data, which will be used in the training process of language model.

### Step2: Train a language model <br />

In this step, we will train a language model based on the responses for the seq2seq model (example data `data/*.vi`). You can also train a language model on any other datasets. 

#### step2.1: Preprocess the data <br />

```
cd lm/tool/
python preprocess.py
```

In this step, we will responses for the seq2seq model (example data `data/*.vi`) will be processed into parallel data for the language model. 

After running the preprocessing, the following files are generated in `lm/data/` folder:

* `train.en`
* `train.de`
* `dev.en`
* `dev.de`

### Step3: Train a language model <br />
