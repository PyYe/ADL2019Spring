# How to run

1. Prepare the dataset and pre-trained embeddings (FastText is used here) in `./data`:

```
./data/train.json
./data/valid.json
./data/test.json
./data/crawl-300d-2M.vec
```

2. Preprocess the data
```
cd src/
python make_dataset.py ../data/
```

3. To train, run
*a poooor example model as follow:
```
python train.py ../models/example/
```

*a RNN w/o Attention model as follow:
```
python train.py ../models/rnn/
```

*a RNN w/ Attention model as follow:
```
python train.py ../models/rnn_attention/
```
4. To predict, run
*a poooor example model as follow:
```
python predict.py ../models/example/ --epoch 3
```
*a RNN w/o Attention model as follow:
```
python predict.py ../models/rnn/ --epoch 3
```
*a RNN w/ Attention model as follow:
```
python predict.py ../models/rnn_attention/ --epoch 3
```
where `--epoch` specifies the save model of which epoch to use.
