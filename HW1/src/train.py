import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger, EarlyStopping
from metrics import Recall




def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embedding'], 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding.vectors

    logging.info('loading valid data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        config['model_parameters']['valid'] = pickle.load(f)

    logging.info('loading train data...')
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)

    if config['arch'] == 'ExampleNet':
        #from modules import ExampleNet
        from predictors import ExamplePredictor
        PredictorClass = ExamplePredictor
        predictor = PredictorClass(
        metrics=[Recall()],
        batch_size=128,
        max_epochs=1000000,
        dropout_rate=0.2, 
        learning_rate=1e-3,
        grad_accumulate_steps=1,
        loss='BCELoss', #BCELoss, FocalLoss
        margin=0, 
        threshold=None,
        similarity='MLP', #inner_product, Cosine, MLP
        **config['model_parameters']
    )
    elif config['arch'] == 'RnnNet':
        from predictors import RnnPredictor
        PredictorClass = RnnPredictor
        predictor = PredictorClass(
        metrics=[Recall()],
        batch_size=128,
        max_epochs=1000000,
        dropout_rate=0.2, 
        learning_rate=1e-3,
        grad_accumulate_steps=1,
        loss='BCELoss', #BCELoss, FocalLoss
        margin=0, 
        threshold=None,
        similarity='MLP', #inner_product, Cosine, MLP
        **config['model_parameters']
    )
        
    elif config['arch'] == 'RnnAttentionNet':
        from predictors import RnnAttentionPredictor
        PredictorClass = RnnAttentionPredictor
        predictor = PredictorClass(
        metrics=[Recall()],
        batch_size=128,
        max_epochs=1000000,
        dropout_rate=0.2, 
        learning_rate=1e-3,
        grad_accumulate_steps=1,
        loss='BCELoss', #BCELoss, FocalLoss
        margin=0, 
        threshold=None,
        similarity='MLP', #inner_product, Cosine, MLP
        **config['model_parameters']
    )
    else:
        logging.warning('Unknown config["arch"] {}'.format(config['arch']))
        #logging.info('Saving test to {}'.format(test_pkl_path))
        
    

    if args.load is not None:
        predictor.load(args.load)

    #def ModelCheckpoint(filepath, monitor='loss', verbose=0, mode='min')
    model_checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model.pkl'),
        'Recall@{}'.format(10), 1, 'all'
    )
    metrics_logger = MetricsLogger(
        os.path.join(args.model_dir, 'log.json')
    )
    
    early_stopping = EarlyStopping(
        os.path.join(args.model_dir, 'model.pkl'),
        monitor='Recall@{}'.format(10),
                 verbose=1,
                 mode='max',
                 patience=10
    )
    
    logging.info('start training!')
    predictor.fit_dataset(train,
                          train.collate_fn,
                          [model_checkpoint, metrics_logger, early_stopping])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
