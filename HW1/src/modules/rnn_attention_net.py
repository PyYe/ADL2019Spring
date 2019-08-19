import torch
torch.cuda.manual_seed_all(518)

class RnnAttentionNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(RnnAttentionNet, self).__init__()
        
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
        
        
    def forward(self, context, context_lens, options, option_lens):
        context_out, context_states = self.rnn(context)
        context = context_out.max(1)[0]
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_out, option_states = self.rnn(option)
            option = option_out.max(1)[0]
            
            
            #similarity between context and each option
            if self._similarity=='inner_product':
                logit = ((context - option) ** 2).sum(-1)
            elif self._similarity=='Cosine': 
                logit = torch.nn.CosineSimilarity(dim=1)(context, option)
            elif self._similarity=='MLP': 
                logit = self.similarity(torch.cat((context, option), dim=-1))[:, 0]
            else:
                pass
            
            logits.append(logit)
        
        logits = torch.stack(logits, 1)
        return logits
