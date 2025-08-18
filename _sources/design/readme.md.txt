<h1 align="center">Email Data Generation using Deep Learning</h1>

<div align="center">

Project for deep-learning-based email generator.

[Document](https://plamephiaj.github.io/data-sythesis-research/)

</div>

> For lawful and compliant research use only. Operate in controlled and isolated environments. Any generated content must not be used for dissemination.



## Project Overview
This project aims to design and implement a deep learning framework that leverages both **Diffusion Models** and **Large Language Models (LLMs)** to collaboratively generate "phishing-style" email samples, enabling security researchers to test/evaluate email filters and train anti-phishing detection models.

With the development of deep learning technology, using deep learning—especially advanced large language models—for phishing email detection has become a promising solution. Data is fundamental to driving the progress of deep learning models, but due to the sensitive nature of phishing emails, which often contain a large amount of malicious content and sensitive information, obtaining such data is much more difficult than for other types. Existing open-source phishing email datasets have undergone various anonymization processes, resulting in the loss of original information, and most of the data is outdated and can no longer meet the needs of rapidly evolving phishing strategies. This work aims to use diffusion models and LLMs to generate phishing emails, providing high-quality training data for downstream detection models.

## Design Principles
Built on [Pytorch Lightning](https://www.lightning.ai/) and [Hydra](https://hydra.cc/), adopting a configuration-driven design philosophy to support flexible experiment configuration and management.

- **Modularity**: Model factory, component decoupling, easy maintenance and upgrades.
- **Reproducibility**: Data/model/config versioning, experiment restoration.
- **Portability**: Support rapid migration and deployment across different environments.

## Functional Architecture
```{mermaid}
flowchart TD
    A["Data Source
    (Open source dataset)"] --> B["Data Processing
    Cleaning/Deidentification/Annotation
    Vectorization/Feature Extraction"]
    B --> C1["Diffusion Model: Initial Style Generation"]
    C1 --> C2["LLM: Proofreading Enhancement"]
    C1 --> RAG

    DATABASE["Spear Target Database"] --> RAG
    RAG["Retrieval-Augmented Generation"] --> C2
    C2 --> D["Downstream Model: Training/Evaluation"]
    D --> E["Entropy Feedback"]
    E --> C1
    E --> C2
```

## Core Modules
### Configuration Module
Input: YAML configuration file path
Output: Configuration object
Function: Load YAML configuration file based on Hydra framework, parse and return configuration object.

### Data Cleaning Module
Input: Raw email data
Output: Cleaned email data
Function: Perform preprocessing operations such as cleaning, de-identification, and annotation on the raw email data, and extract feature vectors.

### Data Preparation Module
Input: Cleaned email data
Output: Prepared training data
Function: Format, split, and otherwise process the cleaned email data to prepare it for model training.

### Model Training Module
Input: Prepared training data, model configuration object
Output: Trained model
Function: Train the model using the prepared training data and the Pytorch Lightning framework, save the trained model.

### Logging Module
Function: Monitoring various information during the training process, including loss values, accuracy, etc., and support visual display (TensorBoard, etc.)

### Evaluation Module
Input: Test data, trained model
Output: Evaluation metrics
Function: Evaluate the trained model, including metrics such as accuracy, recall, etc.

### Documentation Module
Function: Generate project-related documentation, including usage instructions, API documentation, etc.

### Testing Module
Function: Perform unit testing and integration testing on the project to ensure the correctness of each module's functionality.
