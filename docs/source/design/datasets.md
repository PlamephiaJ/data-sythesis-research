# Datasets

## Overview

The datasets are collections of data used for training and evaluating the model. It is crucial to have well-structured datasets to ensure the development process is effective and efficient.

## Structure

The datasets are organized into the following directories:

- `raw/`: Contains the raw, unprocessed data.
- `processed/`: Contains the **well-structured** processed data ready for training.
- `scripts/`: Contains scripts for data processing and augmentation.

## Data Sources

Classic datasets:
- [Assassin](https://spamassassin.apache.org/) (Apache Open-Source Dataset, 2005)
- [CEAS_08](https://plg.uwaterloo.ca/~gvcormac/ceascorpus/) (University of Waterloo, 2008)
- [Enron](https://www.cs.cmu.edu/~enron/) (Carnegie Mellon University, 2000 - 2015)
- [Nazario](https://monkey.org/~jose/phishing/) (José Nazario, 2005 - 2024)
- Nigerian (1998 - 2007)
- [TREC](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/) (University of Waterloo, 2007)

Improved datasets:
- [Curated Datasets 1](https://figshare.com/articles/dataset/Phishing_Email_11_Curated_Datasets/24952503/1)
- [Curated Datasets 2](https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952)

The improved datasets are enhanced and optimized based on the classic datasets. In fact, they are still subsets of the classic datasets and do not include more recent phishing patterns.


## Processing

Data processing is completed in two steps. The first step is performed in the datasets directory using scripts, where the raw datasets are cleaned and extracted, and organized into a well-structured, class-balanced format. The second step is carried out in the corresponding dataset's `LightningDataModule`, which handles tokenization, collation, and other operations, ultimately generating a `DataLoader` for model training and testing.