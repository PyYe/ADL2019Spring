import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.
        #print ('predicts:', predicts)
        #print ('batch["labels"]:', batch['labels'])
        #self.n += predicts.shape[0]
        #self.n_corrects +=  int(sum(predicts.max(1)[1]==batch['labels'].max(1)[1]))
        for i, predict in enumerate(predicts):
            self.n += 1
            best_ids = torch.argsort(predict, descending=True)[:self.at]
            correct_answer_id = torch.argmax(batch['labels'][i])
            if correct_answer_id in best_ids:
                self.n_corrects += 1
        
    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
