import torch
torch.cuda.manual_seed_all(518)
import torch.nn.functional as F

import os
import logging
class RnnAttentionNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings, dropout_rate=0.2,
                 similarity='MLP'):
        super(RnnAttentionNet, self).__init__()
        self.hidden_size = 128
        self.num_layers = 1
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        self.context_layer = torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        self.option_layer = torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)
        self.attention_layer = torch.nn.Linear(self.hidden_size * 2, 1, bias=False)
        self.mlp = torch.nn.Linear((self.hidden_size * 2) * 2, self.hidden_size * 2)
        self.attention_rnn = torch.nn.GRU((self.hidden_size * 2) * 5, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        self.similarity = torch.nn.Linear((self.hidden_size * 2) * 2, 1, bias=False)

        self.dropout = torch.nn.Dropout(dropout_rate)
    def forward(self, context, context_lens, options, option_lens):
        context_length = context.size(1)

        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        # context_outs: tensor of shape (batch_size, context_length, hidden_size * 2)
        # context_h_n: tensor of shape (num_layers * 2, batch_size, hidden_size)
        context_outs, context_h_n = self.rnn(context, h_0)

        # context_outs = self.dropout(context_outs)
        # context_key: tensor of shape (batch_size, context_length, hidden_size * 2)
        context_key = self.context_layer(context_outs)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_length = option.size(1)

            # Set initial hidden and cell states 
            h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # context_outs: tensor of shape (batch_size, option_length, hidden_size * 2)
            # context_h_n: tensor of shape (num_layers * 2, batch_size, hidden_size)
            option_outs, option_h_n = self.rnn(option, h_0)

            #option_outs = self.dropout(option_outs)

            # option_key: tensor of shape (batch_size, option_length, hidden_size * 2)
            option_key = self.option_layer(option_outs)
            
            # repeat_context_outs: tensor of shape (batch_size, context_length, hidden_size * 2) -> (batch_size, option_length, context_length, hidden_size * 2)
            expand_context_keys = torch.unsqueeze(context_key, 1).expand(-1, option_length, -1, -1)
            # repeat_option_outs: tensor of shape (batch_size, option_length, hidden_size * 2) -> (batch_size, option_length, context_length, hidden_size * 2)
            expand_option_keys = torch.unsqueeze(option_key, 2).expand(-1, -1, context_length, -1)
            # attention_weight: tensor of shape (batch_size, option_length, context_length, hidden_size * 4) -> (batch_size, option_length, context_length)
            attention_weight = self.attention_layer(torch.tanh(expand_option_keys + expand_context_keys)).squeeze(dim=-1)
            
            # attention_context_w: tensor of shape (batch_size, context_length, hidden_size * 2)
            attention_context_w = torch.bmm(F.softmax(attention_weight, dim=1).transpose(1, 2), option_outs)
            # attention_option_w: tensor of shape (batch_size, option_length, hidden_size *2)
            attention_option_w = torch.bmm(F.softmax(attention_weight, dim=2), context_outs)

            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # attention_context_outs: tensor of shape (batch_size, context_length, hidden_size * 2)
            # attention_context_h_n: tensor of shape (num_layers * 2, batch_size, hidden_size)
            attention_context_outs, attention_context_h_n = self.attention_rnn(
                torch.cat((
                    context_outs, 
                    attention_context_w, 
                    torch.mul(context_outs, attention_context_w), 
                    context_outs - attention_context_w,
                    self.mlp(torch.cat((context_outs, attention_context_w), dim=-1))), 
                    dim=-1), h_1)

            # Max pooling over RNN outputs.
            # attention_context_outs_max: tensor of shape (batch_size, hidden_size * 2)
            attention_context_outs_max = attention_context_outs.max(dim=1)[0]

            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, option.size(0), self.hidden_size).to(option.get_device())

            # Forward propagate RNN
            # attention_option_outs: tensor of shape (batch_size, option_length, hidden_size * 2)
            # attention_option_h_n: tensor of shape (num_layers * 2, batch_size, hidden_size)
            attention_option_outs, attention_option_h_n = self.attention_rnn(
                torch.cat((
                    option_outs, 
                    attention_option_w, 
                    torch.mul(option_outs, attention_option_w), 
                    option_outs - attention_option_w,
                    self.mlp(torch.cat((option_outs, attention_option_w), dim=-1))), 
                    dim=-1), h_1)

            # Max pooling over RNN outputs.
            # attention_option_outs_max: tensor of shape (batch_size, hidden_size * 2)
            attention_option_outs_max = attention_option_outs.max(dim=1)[0]

            # Cosine similarity between context and each option.
            # logit = torch.nn.CosineSimilarity(dim=1)(attention_context_outs_max, attention_option_outs_max)
            logit = self.similarity(torch.cat((attention_context_outs_max, attention_option_outs_max), dim=-1))[:, 0]
            logits.append(logit)

        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits

