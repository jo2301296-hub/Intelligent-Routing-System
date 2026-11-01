# Consumer Complaint Classification & Intelligent Routing System

An NLP-powered application that classifies financial consumer complaints
into one of ten product categories using deep learning and
Transformer-based models. The system also demonstrates how predicted
categories can support intelligent routing to the appropriate
customer-support team.

## Project Overview

Financial institutions receive large volumes of customer complaints.
Manually reviewing and categorizing every complaint can be
time-consuming and inconsistent.

This project uses Natural Language Processing (NLP) to analyze complaint
narratives and predict the most relevant financial product category. It
compares several neural-network architectures with a fine-tuned
DistilBERT model, then provides an interactive interface built with
Gradio.

## Key Features

-   Classifies complaint text into 10 financial product categories.
-   Compares SimpleRNN, LSTM, GRU, and fine-tuned DistilBERT.
-   Displays the predicted category and confidence score.
-   Displays probabilities across all supported categories.
-   Provides an interactive Gradio interface.
-   Includes an intelligent-routing concept that maps predicted
    categories to support teams.

> **Scope note:** The current classifier predicts one primary category
> per complaint (single-label classification). The routing component is
> a category-to-team mapping demonstration; it does not connect to a
> real bank's internal ticketing system.

## Dataset

The project uses the **Consumer Complaint Database** published by the
U.S. Consumer Financial Protection Bureau (CFPB), accessed through a
Kaggle dataset.

The original dataset contains complaint records and metadata. For this
project, records with missing or empty complaint narratives were
removed, and the task was limited to the following 10 categories with
sufficient examples:

1.  Bank account or service
2.  Consumer Loan
3.  Credit card
4.  Credit reporting
5.  Debt collection
6.  Money transfers
7.  Mortgage
8.  Payday loan
9.  Prepaid card
10. Student loan

Two very small categories were excluded from model training because they
had too few examples for a reliable comparison.

### Data Preparation

The workflow includes:

-   Selecting complaint narratives and their product labels.
-   Removing records with missing or empty narratives.
-   Reviewing duplicate narratives and removing conflicting labels.
-   Creating text representations for sequential models and DistilBERT.
-   Encoding target labels.
-   Splitting the data into stratified training, validation, and test
    sets (80% / 10% / 10%).

The sequential models use cleaned and lemmatized text. DistilBERT uses
less aggressively processed text to preserve context and natural
language structure.

## Models

Four models were trained and evaluated:

  -----------------------------------------------------------------------
  Model                               Description
  ----------------------------------- -----------------------------------
  SimpleRNN                           Recurrent neural network used as a
                                      baseline sequence model.

  LSTM                                Long Short-Term Memory network
                                      designed to learn longer-term
                                      dependencies.

  GRU                                 Gated Recurrent Unit network with a
                                      simpler gated architecture than
                                      LSTM.

  DistilBERT                          Pretrained Transformer fine-tuned
                                      for the 10-class complaint
                                      classification task.
  -----------------------------------------------------------------------

## Results

The following are the results obtained during the project evaluation on
the held-out test set:

  Model              Accuracy   Macro F1   Weighted F1
  ---------------- ---------- ---------- -------------
  SimpleRNN V2           0.83       0.73          0.84
  LSTM                   0.84       0.76          0.85
  GRU                    0.84       0.75          0.84
  **DistilBERT**     **0.89**   **0.82**      **0.88**

**Best-performing model:** Fine-tuned DistilBERT, with 89% accuracy and
a macro F1-score of 0.82.

Macro F1 is included because the dataset is imbalanced and it gives
equal importance to each class. Weighted F1 reflects performance while
accounting for the number of test examples in each class.

## Application Demo

The Gradio application allows a user to:

1.  Enter a financial complaint narrative.
2.  Submit the text for classification.
3.  View the predicted complaint category.
4.  Review the model confidence score.
5.  Explore the probability distribution across categories.

The application uses the saved DistilBERT model and tokenizer.

## Intelligent Routing Concept

The predicted category can be mapped to a support team. Example mappings
include:

  Predicted Category        Suggested Support Team
  ------------------------- ------------------------
  Bank account or service   Banking Support
  Consumer Loan             Consumer Loan Support
  Credit card               Credit Card Support
  Credit reporting          Credit Reporting Team
  Debt collection           Debt Collection Team
  Money transfers           Money Transfer Support
  Mortgage                  Mortgage Support
  Payday loan               Payday Loan Support
  Prepaid card              Prepaid Card Support
  Student loan              Student Loan Support

These are illustrative routing labels and can be customized to match an
organization's actual departments and workflow.

## Technologies Used

-   **Language:** Python
-   **Data processing:** Pandas, NumPy
-   **Machine learning:** Scikit-learn
-   **Deep learning:** TensorFlow / Keras, PyTorch
-   **NLP and Transformers:** Hugging Face Transformers
-   **Pretrained model:** DistilBERT (`distilbert-base-uncased`)
-   **Interface:** Gradio
-   **Development environment:** Kaggle Notebooks

## Repository Structure

A suggested repository structure is:

``` text
Consumer-Complaint-Classification-and-Intelligent-Routing/
│
├── app.py
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── distilbert_model/
│   └── label_encoder.pkl
│
└── notebooks/
    └── model_training.ipynb
```

The exact structure may differ depending on which files are uploaded.
The trained model directory should contain the model configuration,
model weights, and tokenizer files saved by Hugging Face.

## Running the Application

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Consumer-Complaint-Classification-and-Intelligent-Routing
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Check the model path

Make sure `app.py` points to the location of the saved
`distilbert_model` directory in the repository.

### 4. Launch Gradio

``` bash
python app.py
```

The application will provide a local Gradio URL. If the app is
configured to create a public share link, Gradio will also display that
link.

## Important Notes

-   The reported metrics are from the project's current experiment and
    may vary if the data split, preprocessing, random seed, model
    configuration, or training environment changes.
-   The confidence score is the model's predicted probability, not a
    guarantee that the prediction is correct.
-   The model is intended as a prototype and decision-support tool, not
    as a replacement for human review.
-   Complaint narratives may contain personal or sensitive information.
    Do not publish private complaint records or credentials in the
    repository.
-   Make sure the dataset source's terms and usage requirements are
    followed.

## Future Improvements

-   Add multi-label classification to identify multiple issues within
    one complaint.
-   Improve performance on underrepresented categories.
-   Add human review for low-confidence predictions.
-   Implement persistent complaint IDs and status tracking.
-   Connect the routing component to a real ticket-management workflow.
-   Add monitoring and evaluation on new, unseen complaints.

## Acknowledgments

This project was developed as part of my AI training journey at
**Instant Software Solutions**.

Special thanks to **Eng. Jana Hatem** and **Sohaila ElSayed** for their
guidance and support.

------------------------------------------------------------------------

**Project:** Consumer Complaint Classification & Intelligent Routing
System\
**Best Model:** Fine-tuned DistilBERT\
**Test Accuracy:** 89%
