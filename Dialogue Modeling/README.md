# How to run

0. Download [dataset](https://www.kaggle.com/c/13262/download-all)

1. Prepare the dataset and pre-trained embeddings in `./data`:

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
* a poooor example model as follow:
```
python train.py ../models/example/
```

* a RNN w/o Attention model as follow:
```
python train.py ../models/rnn/
```

* a RNN w/ Attention model as follow:
```
python train.py ../models/rnn_attention/
```
4. To predict, run  
* a poooor example model as follow:
```
python predict.py ../models/example/ --epoch 3
```
* a RNN w/o Attention model as follow:
```
python predict.py ../models/rnn/ --epoch 3
```
* a RNN w/ Attention model as follow:
```
python predict.py ../models/rnn_attention/ --epoch 3
```
where `--epoch` specifies the save model of which epoch to use, otherwise, using best model.


# Result

* [Kaggle Competition](https://www.kaggle.com/c/adl2019-homework-1) 
	* Public score :  9.40666
	* Private score : 9.42857

# Reference
ADL2019Spring HW1
Dialogue Modeling
* [Homework 1 Website](https://www.csie.ntu.edu.tw/~miulab/s107-adl/A1)
* [Homework 1 Slide](https://docs.google.com/presentation/d/15LCy7TkJXl2pdz394gSPKY-fkwA3wFusv7h01lWOMSw/edit#slide=id.p)

###### tags: `NTU` `ADL` `2019`