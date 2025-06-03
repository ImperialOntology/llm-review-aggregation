# llm-review-aggregation

## Description
This project implements automatic ontology extraction and review aggregation using Large Language Models (LLMs). It processes product reviews to extract key aspects, build product ontologies, and analyze the sentiment and argumentative strength associated with different product features.

## Project Structure
```Python
📦llm-review-aggregation
 ┣ 📂db
 ┃ ┣ 📜connector.py         # Handles database connection and session management.
 ┃ ┗ 📜manager.py           # Provides higher-level functions for interacting with the database.
 ┣ 📂scripts                # Python and shell scripts for each step of ontology extraction.
 ┃ ┣ 📂python_scripts
 ┃ ┗ 📂shell_scripts
 ┣ 📂src
 ┃ ┣ 📂argumentation
 ┃ ┃ ┣ 📂arg_framework    # Implements the argumentation framework logic (DF-QuAD).
 ┃ ┃ ┣ 📂sentiment        # Modules for sentiment analysis of the reviews.
 ┃ ┣ 📂base
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📂json_grammar              # Defines JSON schema or grammars for data handling.
 ┃ ┃ ┣ 📜amazon_load_preprocess.py # Scripts for loading and preprocessing Amazon review data.
 ┃ ┃ ┣ 📜base_load_preprocess.py   # Base class or common functions for data loading and preprocessing.
 ┃ ┃ ┣ 📜disney_data_preprocess.py # Scripts for loading and preprocessing Disney review data.
 ┃ ┃ ┣ 📜llm_judge_prompts.py      # Defines prompts used for the LLM-as-a-judge evaluation.
 ┃ ┃ ┗ 📜n_shot_examples.py        # Contains examples for few-shot learning used in LLM prompts.
 ┃ ┣ 📂llm_judge          # Implements the LLM-as-a-judge evaluation logic.
 ┃ ┣ 📂ontology
 ┃ ┃ ┣ 📂ontology_bert
 ┃ ┃ ┃ ┣ 📂aspects        # Modules for aspect extraction using BERT.
 ┃ ┃ ┃ ┣ 📂base
 ┃ ┃ ┃ ┣ 📂concepts       # Modules for concept extraction using BERT.
 ┃ ┃ ┃ ┣ 📂relations      # Modules for relation extraction using BERT.
 ┃ ┃ ┃ ┣ 📜helpers.py     # Helper functions for BERT-based ontology extraction.
 ┃ ┃ ┃ ┗ 📜phrase_tokenizer.py # Tokenizer specific for phrase processing in BERT models.
 ┃ ┃ ┣ 📂ontology_llm
 ┃ ┃ ┃ ┣ 📂aspects        # Modules for aspect extraction using LLMs.
 ┃ ┃ ┃ ┣ 📂base           # Base classes for LLM-based ontology extraction.
 ┃ ┃ ┃ ┣ 📂concepts       # Modules for concept extraction using LLMs.
 ┃ ┃ ┃ ┗ 📂relations      # Modules for relation extraction using LLMs.
 ┃ ┃ ┣ 📜synset_extractor.py # Module for extracting synonym sets for concepts.
 ┃ ┃ ┣ 📜tree_builder.py    # Module for constructing the ontology tree structure.
 ┃ ┃ ┗ 📜word_vectoriser_base.py # Base class for word vectorization techniques.
 ┃ ┗ 📜constants.py       # Defines project-wide constants.
 ┣ 📂tests
 ┃ ┣ 📂data               # Test data.
 ┃ ┣ 📂integration        # Integration tests.
 ┃ ┣ 📂unit               # Unit tests.
 ┃ ┗ 📜conftest.py        # Pytest configuration file.
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜config-template
 ┣ 📜logger.py
 ┣ 📜pytest.ini          # Pytest configuration file.
 ┣ 📜requirements.txt
 ┗ 📜run.sh              # Main shell script to run the whole ML pipeline.
```

## Prerequisites

- Python 3.12

## Installation
1. Clone the repository from GitLab:
   ```bash
   git clone https://github.com/ImperialOntology/llm-review-aggregation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd llm-review-aggregation
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Clone the data repository (required for the BERT ontology extraction method). This will create a `bert-train-data` directory inside the `llm-review-aggregation` project directory:
   ```bash
   git clone git@gitlab.doc.ic.ac.uk:g24mai05/bert-train-data.git
   ```
6. Run the setup script to download necessary NLTK datasets:
   ```bash
   python scripts/python_scripts/setup.py
   ```
7. Set up your Hugging Face account:
   - Create a free account on [Hugging Face](https://huggingface.co/).
   - Generate a Hugging Face access token by following [this link](https://huggingface.co/settings/tokens). Make sure to save the token securely.
   - Log in to Hugging Face from your terminal using the command:
      ```bash
      huggingface-cli login
      ```
      and follow the prompts to enter your access token.
   - On the [Mistral-7B-Instruct-v0.2 model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), click the "Agree and access the model" button to accept the terms and gain access.

To verify your installation, you can attempt to run the unit and integration tests as described below.

### Unit Tests
To execute the unit tests for the source code, run the following command:
```bash
export PYTHONPATH=. && pytest tests/unit
```
To measure the unit test coverage of the source code, use this command:
```bash
export PYTHONPATH=. && pytest tests/unit --cov src
```
The current unit test coverage for the source code is 45%.

### Integration Tests
Integration tests can be run using either `pytest` or directly with `python`. When using `pytest`, you might need to include the `-v` (verbose) flag to see detailed output.
```bash
export PYTHONPATH=. && pytest tests/integration -v
# OR
export PYTHONPATH=. && python tests/integration/PATH/TO/SPECIFIC/TEST.py
```
**Caution:** Integration tests might take several minutes to complete. For a quicker initial evaluation, consider running the unit tests first.

## Usage
1. Copy the `config-template` file to create your `config` file:
   ```bash
   cp config-template config
   ```
2. Edit the `config` file to set up the necessary variables (see the next section for detailed explanations).
3. The main pipeline is designed to be executed on the [Imperial High Performance Computing (HPC)](https://www.imperial.ac.uk/computational-methods/hpc/). To submit the `run.sh` script as a job on the HPC, use the following command:
   ```bash
   qsub run.sh
   ```
   For comprehensive information on using the HPC, please refer to the [Imperial HPC documentation](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/).

### Setting up the Config File
The `config` file contains various parameters that control the execution of the pipeline. Below is a description of each parameter:

- `DBNAME`: The name of the PostgreSQL database to be used for storing and retrieving data.
- `USER`: The username for connecting to the PostgreSQL database.
- `PASSWORD`: The password for the specified PostgreSQL database user.
- `PORT`: The port number on which the PostgreSQL database server is listening.
- `HOST`: The hostname or IP address of the PostgreSQL database server.
- `GEMINI_API_KEY`: Your API key for accessing the Google Gemini models via the Google AI API.
- `PYTHONPATH`: Specifies the Python import search path. Setting it to `.` ensures that the project's source code is included.
- `ENV_NAME`: The name of the virtual environment used for the project (e.g., `llm-env`).
- `NJOBS`: The number of parallel jobs to use for certain processing steps (if applicable).
- `BERT_OR_LLM`: A flag to determine whether to use BERT-based or LLM-based methods for ontology extraction. Set to `bert` or `llm`.
- `DATA_SOURCE`: Specifies the primary data source to be used (`amazon` or `disney`).
- `DATA_SOURCE_NAME`: A specific identifier or name for the data source being used.
- `DATA_SOURCE_URL`: The URL or location of the data source (if applicable).
- `CATEGORY_NAME`: The specific product category to be processed from the chosen data source. Should be one of the following: Stand Mixers, Games, Televisions, Wrist Watches, Necklaces, and Disneyland (for `disney` data source only).
- `ASPECT_EXTRACTION_NAME`: A descriptive name for the aspect extraction method being used in the current run.
- `ONTOLOGY_EXTRACTION_NAME`: A descriptive name for the ontology extraction method being used in the current run.
- `ARGUMENT_ANALYSIS_NAME`: A descriptive name for the argument analysis method being used in the current run.
- `ASPECT_EXTRACTION_METHOD`: Analogous to the above.
- `ONTOLOGY_EXTRACTION_METHOD`: Analogous to the above.
- `ARGUMENT_ANALYSIS_METHOD`: Analogous to the above.
- `ASPECT_EXTRACTION_DESCRIPTION`: Analogous to the above.
- `ONTOLOGY_EXTRACTION_DESCRIPTION`: Analogous to the above.
- `ARGUMENT_ANALYSIS_DESCRIPTION`: Analogous to the above.
- `NUMBER_OF_REVIEWS`: The maximum number of reviews to sample from the specified data source and category.
- `TOP_K_ASPECTS_TO_KEEP`: The number of top most frequent aspects to retain after the aspect extraction phase.
- `SETUP_ENV`: A boolean flag indicating whether to create a new python virtual environment. Set to `true` or `false`.
- `BATCH_SIZE`: The batch size to be used for processing data with BERT or Mistralai LLM.
- `RETRAIN_SENTIMENT_BERT`: A boolean flag indicating whether to retrain the sentiment analysis BERT model even if it exists in the cache folder. Set to `true` or `false`.
- `RETRAIN_ENTITY_BERT`: A boolean flag indicating whether to retrain the entity extraction BERT model even if it exists in the cache folder. Set to `true` or `false`.
- `RETRAIN_RELATION_BERT`: A boolean flag indicating whether to retrain the relation extraction BERT model even if it exists in the cache folder. Set to `true` or `false`.
- `RUN_JUDGE`: A boolean flag indicating whether to run the LLM-as-a-judge evaluation. Set to `true` or `false`.

## Acknowledgements

This project builds upon and gratefully acknowledges the following prior work and resources:

- The implementation of the LLM ontology extraction is based on the "Automatic Extraction of Ontologies via Large Language Models" 2024 Master's thesis by Esmanda Wong. We also acknowledge the legacy codebase provided with this thesis, accessible at [https://gitlab.doc.ic.ac.uk/ew1723/llm-ontology-construction](https://gitlab.doc.ic.ac.uk/ew1723/llm-ontology-construction).

- The implementation of the BERT ontology extraction is based on the research presented in "Automatic Product Ontology Extraction from Textual Reviews" 2021 (arXiv:2105.10966v1) by Joel Oksanen, Oana Cocarascu, and Francesca Toni.

- The Disneyland review data utilized in this project is sourced from the publicly available dataset on Kaggle, contributed by arushchillar: [https://www.kaggle.com/datasets/arushchillar/disneyland-reviews](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews).

- The Amazon product review data used in this project is sourced from the Hugging Face Hub dataset provided by the McAuley-Lab: [https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
