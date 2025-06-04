import json
import os
import os.path as p
import pandas
import numpy as np
from Utils import save_data


def Tokenize_Input(first_sentence, second_sentence, tokenizer, model_type):
    first_encoded = tokenizer.encode(str(first_sentence), add_special_tokens=False)
    second_encoded = tokenizer.encode(str(second_sentence), add_special_tokens=False)
    
    if model_type != 'xlnet':
        encoded = [tokenizer.cls_token_id] + first_encoded + [tokenizer.sep_token_id] + second_encoded + [tokenizer.sep_token_id]
    else:
        encoded = first_encoded + [tokenizer.sep_token_id] + second_encoded + [tokenizer.sep_token_id] + [tokenizer.cls_token_id]
    
    return encoded


def get_attention_masks(X, tokenizer):
    attention_masks = []

    for sent in X:
        att_mask = [int(token_id != tokenizer.pad_token_id) for token_id in sent]
        att_mask = np.asarray(att_mask)
        attention_masks.append(att_mask)
    
    return np.asarray(attention_masks)


def pad_seq(seq, max_len, pad_idx):
    if len(seq) > max_len:
        sep = seq[-1]
        seq = seq[0:max_len-1]
        seq.append(sep)
    
    while len(seq) != max_len:
        seq.append(pad_idx)
    
    return seq


def create_data_for_bert(output_location, label_to_idx_dict, data_subset, tokenizer, suffix, model_type):
    if not p.exists(output_location):
        os.mkdir(output_location)

    # Tokenize the input sentences
    X = data_subset.apply(lambda x: Tokenize_Input(x['sentence1'], x['sentence2'], tokenizer, model_type), axis=1)    
    X = pandas.Series(X)

    # Find maximum sequence length
    max_len = max(len(x) for x in X)
    print("actual max length:", max_len)
    
    # Limit maximum length to 300 tokens
    if max_len > 300:
        max_len = 300
    print("using max length:", max_len)

    # Pad sequences to uniform length
    X = X.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)
    X = np.array(X.values.tolist())
    
    # Generate attention masks
    att_mask = get_attention_masks(X, tokenizer)

    # Save tokenized data and attention masks
    save_data(X, output_location + 'X_' + suffix + '.pkl')
    save_data(att_mask, output_location + 'att_mask_' + suffix + '.pkl')
    
    print("Data shapes:", X.shape, att_mask.shape)
    
    # Save labels
    y = np.array(data_subset['label'].tolist())
    save_data(y, output_location + 'y_' + suffix + '.pkl')