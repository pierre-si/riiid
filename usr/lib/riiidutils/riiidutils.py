import torch
from torch.nn.utils.rnn import pad_sequence

class RiiidDataset(torch.utils.data.Dataset):
    def __init__(self, x_cat, x_cont, y, sort_sequences=True, max_length=0):
        seq_lengths = [len(el) for el in x_cat]
        if sort_sequences:
            # sequences must be sorted and order reversed to use pack_padded_sequence with enforce_sorted = True
            _, self.cat, self.cont, self.y = map(list, zip(*sorted(zip(seq_lengths, x_cat, x_cont, y), key=lambda tup: tup[0], reverse=True)))
            #map(list, zip(*sorted(zip(x_cat, x_cont, y), key= lambda i: len(i[0]), reverse=True))) # does not seem to work
        else:
            self.cat = x_cat
            self.cont = x_cont
            self.y = y
        self.max_len = max_length

    def __getitem__(self, key):
        #return {'cat': self.cat[key], 'cont': self.cont[key], 'y': self.y[key]}
        return self.cat[key][-self.max_len:], self.cont[key][-self.max_len:], self.y[key], self.cat[key][-self.max_len:].shape[0]
        
    def __len__(self):
        return len(self.cat)

def riiid_collate_fn(batch):
    cat, cont, y, _ = zip(*batch)
    cat = pad_sequence_left([torch.tensor(el) for el in cat], batch_first=True)
    cont = pad_sequence_left([torch.tensor(el, dtype=torch.float) for el in cont], batch_first=True)
    y = torch.tensor(y, dtype=torch.float)
    return {'cat': cat, 'cont': cont, 'y': y}

def riiid_collate_fn_right_padding(batch):
    cat, cont, y, lengths = zip(*batch)
    cat = pad_sequence([torch.tensor(el) for el in cat], batch_first=True)
    cont = pad_sequence([torch.tensor(el, dtype=torch.float) for el in cont], batch_first=True)
    y = torch.tensor(y, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return {'cat': cat, 'cont': cont, 'y': y, 'lengths': lengths}


def pad_sequence_left(sequences, batch_first=False, padding_value=0.0):
    r"""Pad a list of variable length Tensors with ``padding_value`` to the left.
    Copied from rnn.pad_sequence"""
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor

    return out_tensor