import torch
torch.cuda.manual_seed_all(518)
import torch.nn.functional as F
import logging
import os

class RnnNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dropout_rate=0.2,
                 similarity='inner_product'):
        super(RnnNet, self).__init__()
        
        self.hidden_size = 256  
        self.num_layers = 1
        self._similarity = similarity
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_embeddings, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.similarity = torch.nn.Linear((self.hidden_size * 2) * 2, 1, bias=False)
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, context, context_lens, options, option_lens):
        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Rnn w/o attention for context
        context_out, context_states = self.rnn(context, h_0)
        context_out = self.dropout(context_out)
        context = context_out.max(dim=1)[0]
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # Set initial hidden and cell states 
            h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
            
            # Rnn w/o attention for option
            option_out, option_states = self.rnn(option, h_0)
            option_out = self.dropout(option_out)
            option = option_out.max(1)[0]
            
            
            #similarity between context and each option
            if self._similarity=='inner_product':
                logit = ((context - option) ** 2).sum(-1)
            elif self._similarity=='Cosine': 
                logit = torch.nn.CosineSimilarity(dim=1)(context, option)
            elif self._similarity=='MLP': 
                logit = self.similarity(torch.cat((context, option), dim=-1))[:, 0]
            else:
                logging.warning('Unknown self_similarity {}'.format(self._similarity))
                os.system("pause") 
            
            logits.append(logit)
        
        logits = torch.stack(logits, 1)
        #logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
