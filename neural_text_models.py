import torch
import torch.nn as nn
from transformers import *
from transformers.modeling_utils import *


class BERT(nn.Module):
    
    def __init__(self, num_classes):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-cased", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output


class BERT_with_dropout(nn.Module):
    
    def __init__(self, num_classes, bert_dropout):
        super(BERT_with_dropout, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-cased", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(p=bert_dropout)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        output = self.linear(pooled_output)
        return output


class BERT_for_contrastive(nn.Module):
    
    def __init__(self, num_classes):
        super(BERT_for_contrastive, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-cased", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear_contrastive = nn.Linear(768, 1024)
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output_for_contrastive = self.linear_contrastive(pooled_output)
        output = self.linear(pooled_output)
        return output, output_for_contrastive


class BERT_locally_pretrained(nn.Module):
    
    def __init__(self, num_classes, pretrained_model_path):
        super(BERT_locally_pretrained, self).__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_path, 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output


class Sci_BERT(nn.Module):
    
    def __init__(self, num_classes):
        super(Sci_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(
            "allenai/scibert_scivocab_cased", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output


class Sci_BERT_uncased(nn.Module):
    
    def __init__(self, num_classes):
        super(Sci_BERT_uncased, self).__init__()
        self.bert = BertModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output

class Bio_BERT(nn.Module):
    
    def __init__(self, num_classes):
        super(Bio_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output


class XLnet(nn.Module):
    
    def __init__(self, num_classes):
        super(XLnet, self).__init__()
        self.bert = XLNetModel.from_pretrained(
            "xlnet-base-cased", 
            output_attentions=False, 
            output_hidden_states=True,
            return_dict=False
        )
        self.sequence_summary = SequenceSummary(self.bert.config)
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.sequence_summary(output[0])
        output = self.linear(pooled_output)
        return output


class RoBERTa(nn.Module):
    
    def __init__(self, num_classes):
        super(RoBERTa, self).__init__()
        self.bert = RobertaModel.from_pretrained(
            "roberta-base", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output


class RoBERTa_large(nn.Module):
    """Large RoBERTa model for text classification."""
    
    def __init__(self, num_classes):
        super(RoBERTa_large, self).__init__()
        self.bert = RobertaModel.from_pretrained(
            "roberta-large", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(1024, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output


class RoBERTa_large_for_contrastive(nn.Module):
    """Large RoBERTa model with contrastive learning output."""
    
    def __init__(self, num_classes):
        super(RoBERTa_large_for_contrastive, self).__init__()
        self.bert = RobertaModel.from_pretrained(
            "roberta-large", 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=False
        )
        self.linear = nn.Linear(1024, num_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        output = self.linear(pooled_output)
        return output, pooled_output