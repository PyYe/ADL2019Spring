import torch
torch.cuda.manual_seed_all(518)
from .base_predictor import BasePredictor
import sys
sys.path.append('..')
from modules import RnnAttentionNet

from focalloss import FocalLoss
import logging

class RnnAttentionPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding,
                 dropout_rate=0.2, loss='FocalLoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(RnnAttentionPredictor, self).__init__(**kwargs)
        self.model = RnnAttentionNet(embedding.size(1),
                                similarity=similarity)
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

        # use cuda
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'BCELoss': torch.nn.BCEWithLogitsLoss(),
            'FocalLoss': FocalLoss(),
            #--
            'L1Loss': torch.nn.L1Loss(),
            'SmoothL1Loss': torch.nn.SmoothL1Loss(),
            'MSELoss': torch.nn.MSELoss(),
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
            'NLLLoss': torch.nn.NLLLoss(),
            #'NLLLoss2d': torch.nn.NLLLoss2d(),
            'KLDivLoss': torch.nn.KLDivLoss(),
            'MarginRankingLoss': torch.nn.MarginRankingLoss(),
            'MultiMarginLoss': torch.nn.MultiMarginLoss(),
            'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss(),
            'SoftMarginLoss': torch.nn.SoftMarginLoss(),
            'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss(),
            'CosineEmbeddingLoss': torch.nn.CosineEmbeddingLoss(),
            'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss()
            #'TripleMarginLoss': torch.nn.TripleMarginLoss()                
        }[loss]
        logging.info('Loss using ' + loss + '...')
        logging.info('Similarity using ' + similarity + '...')
        
    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        return logits
