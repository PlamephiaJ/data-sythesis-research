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
- [Assassin](https://spamassassin.apache.org/old/publiccorpus) (Apache Open-Source Dataset, 2005)
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

## Dataset-specific Information
> WARNING! Corpus may contain viruses, fraudulent solicitations, and other files that may pose a security risk.  Do not view any files in the folder with an ordinary browser or email client.  Also note that virus or adware removal tools may damage the corpus.
### Classic Datasets
Kaggle all-in-one solution [link](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset).
#### [Assassin](https://spamassassin.apache.org/old/publiccorpus) (Apache Open-Source Dataset, 2005)
This dataset is contributed by the internet community and is one of the earliest spam email datasets, covering a variety of topics.
- spam: 500 spam messages, all received from non-spam-trap sources.

- easy_ham: 2500 non-spam messages.  These are typically quite easy to differentiate from spam, since they frequently do not contain any spammish signatures (like HTML etc).

- hard_ham: 250 non-spam messages which are closer in many respects to typical spam: use of HTML, unusual HTML markup, coloured text, "spammish-sounding" phrases etc.

- easy_ham_2: 1400 non-spam messages.  A more recent addition to the set.

- spam_2: 1397 spam messages.  Again, more recent.

Total count: 6047 messages, with about a 31% spam ratio.
#### [CEAS_08](https://plg.uwaterloo.ca/~gvcormac/ceascorpus/) (University of Waterloo, 2008)
This dataset comes from the CEAS 2008 email detection challenge. All emails during the competition were collected in real time and cover a wide range of topics.
The structure of the original dataset is relatively complex. You can use the existing CEAS_08 dataset available on Kaggle, such as [link](https://www.kaggle.com/datasets/doryanay/ceas-08). 39154 samples in total, 17312 (44.2%) benign, 21842 (55.8%) spam.
#### [Enron](https://www.cs.cmu.edu/~enron/) (Carnegie Mellon University, 2000 - 2015)
This dataset is a collection of emails from the Enron Corporation, which was involved in a major corporate scandal.
Kaggle version can be found at [link](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset). 517401 samples in total.
#### [Nazario](https://monkey.org/~jose/phishing/) (José Nazario, 2005 - 2024)
This dataset is a collection of phishing emails curated by José Nazario over nearly two decades. Its most notable feature is that it is entirely constructed from phishing emails received in his personal inbox.
#### Nigerian (1998 - 2007)
These "Nigerian" datasets typically focus on "Nigerian fraud emails," that is, phishing and scam emails, rather than the more common advertising or junk mail types found in spam/ham classification.
#### [TREC](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/) (University of Waterloo, 2007)

### Improved Datasets
#### [Curated Datasets 1](https://figshare.com/articles/dataset/Phishing_Email_11_Curated_Datasets/24952503/1)
#### [Curated Datasets 2](https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952)
