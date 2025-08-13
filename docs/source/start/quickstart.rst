.. _quickstart:

=============================================================
Quickstart: BERT Pre-trained Model Fine-tuning on WAF Dataset
=============================================================

This guide demonstrates how to fine-tune a BERT pre-trained model on a Web Application Firewall (WAF) dataset for malicious request classification.

Introduction
------------

.. _bert_uncased_L-2_H-128_A-2: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2

In this example, we fine-tune the compact ``bert_uncased_L-2_H-128_A-2`` model from Hugging Face to classify HTTP requests as malicious or benign using a WAF dataset. This process leverages the ``smf`` framework for efficient model training. For background, refer to [1]_.

Prerequisites:

- The latest version of the ``smf`` framework and its dependencies, installed per the `installation guide <https://github.com/smf/smf-install>`_. Using the provided Docker image is recommended.
- A GPU with at least 16 GB HBM (e.g., NVIDIA RTX3090 or RTX4090).
- Python 3.9+ and Conda installed.
- Access to the WAF dataset repository on CodeHub.

Dataset Introduction
--------------------

The WAF dataset contains HTTP request logs labeled as malicious or benign, designed for training models to detect web-based attacks (e.g., SQL injection, XSS). It includes features such as URL paths, headers, and payloads, preprocessed into a format suitable for BERT's input pipeline. The dataset is hosted on the CodeHub dataset repository and consists of approximately 50,000 samples, split into 80% training and 20% validation sets.

Step 1: Prepare the Dataset
---------------------------

Clone the WAF dataset from the CodeHub repository and initialize submodules.

.. code-block:: bash

   git clone https://codehub.example.com/waf-dataset.git
   cd waf-dataset
   git submodule init
   git submodule update

The dataset will be downloaded to the ``data/`` directory in CSV format, with columns for request text and labels (0 for benign, 1 for malicious).

Step 2: Download the Base Model
-------------------------------

Download the pre-trained ``bert_uncased_L-2_H-128_A-2`` model from Hugging Face.

.. code-block:: bash

   mkdir -p models
   cd models
   git lfs install
   git clone https://huggingface.co/google/bert_uncased_L-2_H-128_A-2

The model weights and configuration will be stored in ``models/bert_uncased_L-2_H-128_A-2/``.

Step 3: Fine-tune the Model
---------------------------

Fine-tune the model using a configuration file tailored for the WAF dataset.

1. Ensure the ``smf_env`` Conda environment is activated:

.. code-block:: bash

   conda activate smf_env

2. Navigate to the ``smf`` source directory and run the training script with the provided configuration:

.. code-block:: bash

   cd smf/src
   python main.py --config ai_waf/bert_L2H128A2.yaml

Sample configuration file (``ai_waf/bert_L2H128A2.yaml``):

.. code-block:: yaml

   model:
     pretrained_path: ../models/bert_uncased_L-2_H-128_A-2
     num_labels: 2
   dataset:
     path: ../../waf-dataset/data
     train_split: train.csv
     val_split: val.csv
     max_length: 128
   training:
     batch_size: 32
     learning_rate: 2e-5
     epochs: 3
     optimizer: adamw
     scheduler: linear
   output:
     save_dir: ../outputs/waf_finetuned
     log_interval: 100

This configuration specifies the model path, dataset details, and training hyperparameters. Adjust paths as needed based on your directory structure.

Step 4: Evaluate the Model
--------------------------

After training, evaluate the model on the validation set:

.. code-block:: bash

   python evaluate.py --model ../outputs/waf_finetuned --data ../../waf-dataset/data/val.csv

The script outputs metrics such as accuracy, precision, recall, and F1-score.

References
----------

.. [1] Well-Read Students Learn Better: On the Importance of Pre-training Compact Models (https://arxiv.org/abs/1908.08962). This paper highlights the benefits of pre-training compact models like BERT for downstream tasks.
