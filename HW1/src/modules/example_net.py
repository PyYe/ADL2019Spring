import torch
torch.cuda.manual_seed_all(518)
import torch.nn.functional as F

class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_embeddings, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )

    def forward(self, context, context_lens, options, option_lens):
        #print ('context_lens:', context_lens)#[29, 27, 33, 4, 26, 33, 8, 18, 3, 9] batch_size:10
        #print ('option_lens:', option_lens)#[[9, 8, 3, 18, 3],.. , [12, 27, 3, 2, 18]] ten(batch_size) [] in [] (9, 8, 3, 18, 3 are every options(1 positive; 4 negative) lengths at batch 1) 
        #print ('context_:', context)
        #print ('context_ shape:', context.shape) #[batch_size(10), padded_len_of_batch(52/33...), dim_word_vector(300)]
        #print ('_context_', self.mlp(context))
        #print ('_context_.shape', self.mlp(context).shape) #[batch_size(10), padded_len_of_batch(52/33...), dim_mlp_output(256)]
        context = self.mlp(context).max(1)[0] # torch.tensor([[[1,2,3],[3,2,1]],[[7,8,9],[10,11,12]]]) -> tensor([[ 3,  2,  3], [10, 11, 12]])
        #print ('_context:', context)
        #print ('_context shape:', context.shape) #[batch_size(10), dim_mlp_output(256)] dim_mlp_output(256) means a context vector
        logits = []
        #print ('options_ shape:', options.shape) #options_ shape: torch.Size([batch_size(10), num_options(5), padded_len_of_option_batch(50), dim_word_vector(300)])
        #print ('_options shape:', options.transpose(1, 0).shape) #_options shape: torch.Size([ num_options(5), batch_size(10), padded_len_of_option_batch(50), dim_word_vector(300)])
        for i, option in enumerate(options.transpose(1, 0)):
            #if i == 0:
                #print ('option_:', option)
                #print ('option_ shape:', option.shape) #[batch_size(10), padded_len_of_batch(43/50...), dim_word_vector(300)]
            option = self.mlp(option).max(1)[0]
            #if i == 0:    
                #print ('_option:', option)
                #print ('_option shape:', option.shape) #[batch_size(10), dim_mlp_output(256)]
            logit = ((context - option) ** 2).sum(-1) # logit represents the similarity between context and a certain option
            #if i == 0:
                #print ('logit:', logit)  #logit: tensor([0.5173, 0.7309, 0.6070, 0.6796, 0.2928, 0.9204, 0.1908, 0.3425, 0.5297, 0.4618], device='cuda:0', grad_fn=<SumBackward2>)

                #print ('logit shape:', logit.shape)  logit shape: torch.Size([10]) (batch_size)
            logits.append(logit)
        #logits = torch.stack(logits, 1)
        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
