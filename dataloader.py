import torch
import spacy
import random
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer, to_map_style_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def yiled_token(dataset, tokenizer):
    index = 0 if type(tokenizer.keywords['spacy']) == spacy.lang.de.German else 1
    for items in dataset:
        yield tokenizer(items[index].lower())

def transform2token(dataset, tokenizer_src, tokenizer_trg, vocab_src, vocab_trg):
    length = len(dataset)
    for i in range(length):
        src, trg = dataset._data[i]
        src = tokenizer_src(src)
        trg = tokenizer_trg(trg)
        src = [vocab_src['<bos>']] + [vocab_src[x.lower()] for x in src] + [vocab_src['<eos>']]
        trg = [vocab_trg['<bos>']] + [vocab_trg[x.lower()] for x in trg] + [vocab_trg['<eos>']]
        dataset._data[i] = (torch.LongTensor(src), torch.LongTensor(trg))
    return dataset

def collate_batch(batch):
    src_list, trg_list = [], []
    for src, trg in batch:
        src_list.append(src)
        trg_list.append(trg)
    return pad_sequence(src_list), pad_sequence(trg_list)

def batch_sample(dataset, batch_size):
    indices = [(i, len(s[0])) for i, s in enumerate(dataset)]
    while True:
        random.shuffle(indices)
        pooled_indices = []
        for i in range(0, len(indices), batch_size*100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
        pooled_indices = [x[0] for x in pooled_indices]
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]


def get_tokenizer_and_vocab(dataset):
    tokenizer_de, tokenizer_en = get_tokenizer('spacy','de_core_news_sm'), get_tokenizer('spacy','en_core_web_sm')
    vocab_de = build_vocab_from_iterator(yiled_token(dataset, tokenizer_de), min_freq=2, specials=['<pad>','<unk>','<bos>','<eos>'])
    vocab_en = build_vocab_from_iterator(yiled_token(dataset, tokenizer_en), min_freq=2, specials=['<pad>','<unk>','<bos>','<eos>'])
    vocab_de.set_default_index(1)
    vocab_en.set_default_index(1)
    return tokenizer_de, tokenizer_en, vocab_de, vocab_en

def get_dataloader_and_etc(train_dataset, val_dataset, test_dataset, batch_size = 8):
    train_dataset, val_dataset, test_dataset = to_map_style_dataset(train_dataset),to_map_style_dataset(val_dataset), to_map_style_dataset(test_dataset)
    tokenizer_de, tokenizer_en, vocab_de, vocab_en = get_tokenizer_and_vocab(train_dataset)
    train_dataset = transform2token(train_dataset, tokenizer_de, tokenizer_en, vocab_de, vocab_en)
    val_dataset = transform2token(val_dataset, tokenizer_de, tokenizer_en, vocab_de, vocab_en)
    test_dataset = transform2token(test_dataset, tokenizer_de, tokenizer_en, vocab_de, vocab_en)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sample(train_dataset, batch_size), collate_fn=collate_batch)
    val_dataloader = DataLoader(test_dataset, batch_sampler=batch_sample(test_dataset, batch_size), collate_fn=collate_batch)
    test_dataloader = DataLoader(val_dataset, batch_sampler=batch_sample(val_dataset, batch_size), collate_fn=collate_batch)
    return train_dataloader, val_dataloader, test_dataloader,(tokenizer_de, tokenizer_en, vocab_de, vocab_en)
