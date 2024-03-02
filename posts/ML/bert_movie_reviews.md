---
layout: default
title:  IMDB movie reviews
parent: Machine Learning
has_children: False
nav_order: 1
---

The goal of this notebook is to understand some details about Huggingface's *dataset* and *transformer* libraries and also as a reference point for fine-tuning a LLM model for a classification task.


```python
!pip install -q transformers datasets

```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m510.5/510.5 kB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m13.1 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
# For NN development, we will be using Pytorch-Lightning package
!pip install lightning --quiet
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.1/2.1 MB[0m [31m7.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m840.4/840.4 kB[0m [31m8.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m800.9/800.9 kB[0m [31m9.8 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import numpy as np
import torch
import lightning as L

from sklearn import metrics
from collections import Counter
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from torch import nn, optim
from torchmetrics.classification import BinaryAccuracy
```


```python
# Setting up the device for GPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

    cuda


## Data


```python
# Inspect the dataset
ds_builder = load_dataset_builder("imdb")
print(ds_builder.info.description)
print(ds_builder.info.features)
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    Downloading readme:   0%|          | 0.00/7.81k [00:00<?, ?B/s]


    
    {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}



```python
# Let's inspect the splits
get_dataset_split_names("imdb")
```




    ['train', 'test', 'unsupervised']




```python
# Now let's load the train and the test dataset splits
dataset_train = load_dataset(path="imdb", split="train")
dataset_test = load_dataset(path="imdb", split="test")
```


    Downloading data:   0%|          | 0.00/21.0M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/20.5M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/42.0M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/25000 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/25000 [00:00<?, ? examples/s]



    Generating unsupervised split:   0%|          | 0/50000 [00:00<?, ? examples/s]



```python
# Let's now define some key variables
TRAIN_SAMPLE_SIZE = 1000
TEST_SAMPLE_SIZE = 1000
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-5
MAX_LEN = 200
```


```python
# For the sake of this tutorial I will subsample to a size of 1000
train_indices = np.random.choice(len(dataset_train),
                                 size=TRAIN_SAMPLE_SIZE, replace=False)
data_train = dataset_train.select(train_indices)

test_indices = np.random.choice(len(dataset_test),
                                size=TEST_SAMPLE_SIZE, replace=False)
data_test = dataset_test.select(test_indices)
```


```python
# Let's make sure we have good distribution of class labels
from collections import Counter

Counter(data_train['label'])
```




    Counter({0: 513, 1: 487})



## Toknenization

We can now start to *tokenize* the input data.


```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



```python
# We apply the BERT tokenizer with a max_len of 200 and padding or truncating
# text if needed. Since each `review` is a single entity, we don't require the
# token_type_ids which should all be just 0's.
def tokenization(datarow):
  return tokenizer(datarow["text"], max_length=MAX_LEN,
                   return_token_type_ids=False, padding='max_length',
                   truncation=True)
```


```python
# As per this (https://huggingface.co/docs/datasets/en/use_dataset),
# this is a good way to apply tokenization to the entire dataset.
data_train = data_train.map(tokenization, batched=True)
data_test = data_test.map(tokenization, batched=True)
```


    Map:   0%|          | 0/1000 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1000 [00:00<?, ? examples/s]


The dataset returns lists in the output. However, we would like to use it in Pytorch and want it to return `tensors` instead. To do this, we will set its
format as described [here](https://huggingface.co/docs/datasets/en/use_with_pytorch)


```python
data_train = data_train.with_format("torch", device=device)
data_test = data_test.with_format("torch", device=device)
```


```python
# Remove unwanted column 'text' since we have already tokenized it.
data_train = data_train.remove_columns(column_names=["text"])
data_test = data_test.remove_columns(column_names=["text"])
```

We can pass these Huggingface dataset directly to the Torch Dataloaders.


```python
train_params = {
    "batch_size": TRAIN_BATCH_SIZE,
    "shuffle": True,
}

test_params = {
    "batch_size": TEST_BATCH_SIZE,
    "shuffle": False
}

train_dataloader = DataLoader(data_train, **train_params)
test_dataloader = DataLoader(data_test, **test_params)
```

## Model


```python
class FineTuneBert(L.LightningModule):
    def __init__(self, bert_backbone) -> None:
        super().__init__()
        self.bert_backbone = bert_backbone
        self.dropout = nn.Dropout(p=0.3)

        # In the loss function, there are multiple ways we can go:
        # 1. Use CrossEntropyLoss, in which case the out_features = 2
        # 2. Use BCELoss, in which case the out_features = 1, but we need an additional
        #    sigmoid layer at the end.
        # 3. BCEWithLogitsLoss, in which case the out_features = 1, but we do not need
        #    an additional sigmoid layer - WE WILL BE USING THIS.
        self.fc1 = nn.Linear(in_features=768, out_features=1)

    def forward(self, input_ids, attention_mask):
        # pass the inputs through the backbone
        # The output of BERT is last_hidden_state and pooler_output. Here we are
        # concerned about the latter.
        _, bert_output = self.bert_backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.dropout(bert_output)
        output = self.fc1(output)
        output = output.view(-1)
        return output

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def training_step(self, batch, batch_idx):
        label, input_ids, attention_mask = (
            batch['label'],
            batch['input_ids'],
            batch['attention_mask']
        )

        preds = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, label.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        label, input_ids, attention_mask = (
            batch['label'],
            batch['input_ids'],
            batch['attention_mask'],
        )
        preds = self(input_ids, attention_mask)
        test_loss = nn.functional.binary_cross_entropy_with_logits(preds, label.float())
        self.log("test_loss", test_loss, prog_bar=True)

        # Compute accuracy
        acc_metric = BinaryAccuracy().to(device)
        self.log("test_acc", acc_metric(preds, label), on_epoch=True)
```


```python
bert = AutoModel.from_pretrained("bert-base-uncased")

model = FineTuneBert(bert_backbone=bert)
model.to(device)
```




    FineTuneBert(
      (bert_backbone): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0-11): 12 x BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (dropout): Dropout(p=0.3, inplace=False)
      (fc1): Linear(in_features=768, out_features=1, bias=True)
    )




```python
trainer = L.Trainer(max_epochs=5)
trainer.fit(model=model, train_dataloaders=train_dataloader)
```

    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: IPU available: False, using: 0 IPUs
    INFO:lightning.pytorch.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name          | Type      | Params
    --------------------------------------------
    0 | bert_backbone | BertModel | 109 M 
    1 | dropout       | Dropout   | 0     
    2 | fc1           | Linear    | 769   
    --------------------------------------------
    109 M     Trainable params
    0         Non-trainable params
    109 M     Total params
    437.932   Total estimated model params size (MB)
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name          | Type      | Params
    --------------------------------------------
    0 | bert_backbone | BertModel | 109 M 
    1 | dropout       | Dropout   | 0     
    2 | fc1           | Linear    | 769   
    --------------------------------------------
    109 M     Trainable params
    0         Non-trainable params
    109 M     Total params
    437.932   Total estimated model params size (MB)



    Training: |          | 0/? [00:00<?, ?it/s]


    INFO: `Trainer.fit` stopped: `max_epochs=5` reached.
    INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=5` reached.



```python
trainer.test(model=model, dataloaders=test_dataloader)
```

    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]



    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_acc          </span>│<span style="color: #800080; text-decoration-color: #800080">    0.8849999904632568     </span>│
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.39432957768440247    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>






    [{'test_loss': 0.39432957768440247, 'test_acc': 0.8849999904632568}]




```python

```
