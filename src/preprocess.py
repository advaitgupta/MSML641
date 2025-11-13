import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

VOCAB_SIZE = 10000 
UNK_TOKEN, PAD_TOKEN = "<unk>", "<pad>"
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN]

tokenizer = get_tokenizer('basic_english')

def build_vocabulary(train_iter):
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter),
        max_tokens=VOCAB_SIZE - len(SPECIAL_TOKENS), 
        specials=SPECIAL_TOKENS
    )
    vocab.set_default_index(vocab[UNK_TOKEN])
    return vocab

def collate_batch(batch, vocab, max_seq_len):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(1.0 if _label == 2 else 0.0) 
        
        tokens = tokenizer(_text)
        token_ids = vocab(tokens)
        
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len] 
        
        text_list.append(torch.tensor(token_ids, dtype=torch.int64))

    pad_idx = vocab[PAD_TOKEN]
    padded_texts = pad_sequence(text_list, 
                                batch_first=True, 
                                padding_value=pad_idx)
    
    if padded_texts.shape[1] < max_seq_len:
        padding = torch.full(
            (padded_texts.shape[0], max_seq_len - padded_texts.shape[1]),
            pad_idx,
            dtype=torch.int64
        )
        padded_texts = torch.cat((padded_texts, padding), dim=1)
    
    labels = torch.tensor(label_list, dtype=torch.float32)
    return labels.unsqueeze(1), padded_texts

def get_dataloaders(max_seq_len, batch_size):
    train_iter, test_iter = IMDB(split=('train', 'test'))
    
    train_iter_for_vocab, train_iter_for_loader = IMDB(split=('train', 'train'))
    vocab = build_vocabulary(train_iter_for_vocab)
    
    collate_fn = lambda batch: collate_batch(batch, vocab, max_seq_len)
    
    train_dataloader = DataLoader(
        list(train_iter_for_loader),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        list(test_iter),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_dataloader, test_dataloader, len(vocab)