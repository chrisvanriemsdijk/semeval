
# Codebase: Improving performance for SemEval2019 with modern NLP techniques

This project is the codebase of the research 'Improving performance for SemEval2019 with modern NLP techniques' of November 2023 at the University of Groningen. Aimed at classifying of offensive tweets. Both, concerning classic models and more advanced models.

How to download all required data and packages can be found in section [Run](#run)
## File structure

```
CODEBASE
|-- data
|   |-- dev.tsv
|   |-- GPT.csv
|   |-- test.tsv
|   |-- train.tsv
|-- helpers
|   |-- helpers_baseline.py
|   |-- helpers_general.py
|   |-- helpers_pretrained.py
|   |-- helpers_lstm.py
|-- results
|   |-- ...
|-- .gitignore
|-- requirements.txt
|-- run_baseline.py
|-- run_lstm.py
|-- run_pretrained.py
```


### data
The data directory consists of four data files. All train, dev and test data is contained in the corresponding file. Examples which are generated by GPT are stored in `GPT.csv`. This file contains the samples which are used to obtain the  reported results.

Any results will be saved in the `results` directory.
### helpers
In total there exist four helper files. The helperfiles consist of code needed by the corresponding models. `helpers_general` consists of code that is used by two or more models. The functions are called when running one of the models and can not be ran independently.


### models
Various are provided. Divided into three seperate files. Classic models with and without POS can be found in `run_baseline.py`. LSTM model can be found in `run_lstm.py`. And BERT models can be found in `run_pretrained.py`.

#### run_baseline
Firstly, the data will be undersampled after which it will train and test each model for each n-gram range, the ranges to be tested can be defined. 
Which models should be tested can be selected in the function `run_range`. The array `classifiers` contains all tested classifiers, KNN and SVM. If you do not want to run all classifiers, it is possible to comment out some of them. 

If wanted to run with POS, please use the corresponding flag. All possible flags:
```
--train_file {-> Input file to learn from
--dev_file {-> Separate dev set to read in
--test_file {-> If added, use trained model to predict on test set
--results_dir {-> Where to store results
--part_of_speech {-> Define whether to use POS
--rangestart {-> Which minimum n-gram to test
--rangeend {-> Which maximum n-gram to test
--min_df {-> Minimum appearance of n-grams
--stem {-> Add flag if should be stemmed
--lemmatize {-> Add flag if should be lemmatized
--emoji_remove {-> Add flag if emojis should be removed
```

#### run_lstm
TODO

#### run_pretrained
Firstly, the data will be undersampled after which it will train and test each model.
Which pretrained models should be tested can be selected in the main function. The dict `pretrained` contains all tested models. If you do not want to run all models, it is possible to comment out some of them. 

All possible flags:
```
--train_file {-> Input file to learn from
--dev_file {-> Separate dev set to read in
--test_file {-> If added, use trained model to predict on test set
--gpt_file {-> If added, use gpt generated samples to train
--results_dir {-> Where to store results
--stem {-> Add flag if should be stemmed
--lemmatize {-> Add flag if should be lemmatized
--emoji_remove {-> Add flag if emojis should be removed
--epochs {-> Set number of epochs
--batch {-> Set batch size
--startrate {-> Set start of polynomial learning rate
--endrate {-> Set end of polynomial learning rate
--seqlen {-> Set sequence length
```
## Run
### Pre requisites
--TO DO

### Run
The models can be run via their corresponding run file, add flags if wanted. For instance for the baseline with part of speech tagging.
```
python3 run_baseline.py --part_of_speech
```

The data will be taken from the data directory and results are stored in the results directory.