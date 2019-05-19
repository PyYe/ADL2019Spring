import json
import logging
from multiprocessing.dummy import Pool
from dataset import DialogDataset
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec
class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)

    def tokenize(self, sentence):#sentence to list of words.
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence. 
        """
        # TODO
        return list(gensim.utils.tokenize(sentence))
        pass

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        # TODO
        # Hint: You can use `self.embedding`
        train_sequences = []
        #print ('list(gensim.utils.tokenize(sentence))', list(gensim.utils.tokenize(sentence)))
        for i, s in enumerate(list(gensim.utils.tokenize(sentence))):
            #if i == 0:
            #    print ('s:',s)
            toks = self.embedding.to_index(s)   # Plus 1 to reserve index 0 for OOV words
            #if i == 0:
            #    print ('toks:',toks)
            train_sequences.append(toks)
        print ('train_sequences', train_sequences)
        return train_sequences
        pass

    def collect_words(self, data_path, n_workers=4):
        with open(data_path) as f:
            data = json.load(f)

        utterances = []
        for sample in data:
            utterances += (
                [message['utterance']
                 for message in sample['messages-so-far']]
                + [option['utterance']
                   for option in sample['options-for-next']]
            )
        utterances = list(set(utterances)) #['str1', 'str2', ...] (unique)
        chunks = [
            ' '.join(utterances[i:i + len(utterances) // n_workers]) #['str1', 'str2'] --> 'str1 str2'
            for i in range(0, len(utterances), len(utterances) // n_workers)
        ]
        #print ('chunks',chunks)
        with Pool(n_workers) as pool:
            chunks = pool.map_async(self.tokenize, chunks)
            words = set(sum(chunks.get(), []))

        #words =set()
        return words

    def get_dataset(self, data_path, n_workers=4, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        self.logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = json.load(f)

        self.logging.info('preprocessing data...')

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])

                # When debugging, you'd better not use multi-thread.
                # results[i] = self.preprocess_dataset(batch, preprocess_args)

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()

        padding = self.embedding.to_index('</s>')
        return DialogDataset(processed, padding=padding, **dataset_args)

    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset, ascii=True):
            processed.append(self.preprocess_sample(sample))

        return processed

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['id'] = data['example-id']

        # process messages-so-far; processed['context'] processed['speaker']
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']:
            processed['context'].append(
                self.sentence_to_indices(message['utterance'].lower())
            )

        ## process options
        processed['options'] = []
        processed['option_ids'] = []

        # process correct options
        if 'options-for-correct-answers' in data:
            processed['n_corrects'] = len(data['options-for-correct-answers'])
            for option in data['options-for-correct-answers']:
                processed['options'].append(
                    self.sentence_to_indices(option['utterance'].lower())
                )
                processed['option_ids'].append(option['candidate-id'])
        else:
            processed['n_corrects'] = 0

        # process the other options
        for option in data['options-for-next']:
            if option['candidate-id'] in processed['option_ids']:
                continue

            processed['options'].append(
                self.sentence_to_indices(option['utterance'].lower())
            )
            processed['option_ids'].append(option['candidate-id'])

        return processed
