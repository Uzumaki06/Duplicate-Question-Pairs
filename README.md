This project aims to detect duplicate question pairs using natural language processing (NLP) techniques and machine learning algorithms. Duplicate question detection is essential in many applications such as customer support, forums, and Q&A platforms like Quora. This project involves creating a model to identify whether two questions are semantically equivalent.

# Key Objectives

1. Identify Duplicate Questions: Develop a system that can automatically detect pairs of questions that have the same intent or meaning.
2. Improve User Experience: Help users find answers more quickly by redirecting them to existing questions rather than creating new ones.
3. Enhance Data Management: Streamline the data by consolidating duplicate questions, which can help in managing the content more effectively.

# Steps Involved

1.Data Collection and Preprocessing:
Gather a dataset containing pairs of questions labeled as duplicates or not duplicates.
Preprocess the data to remove noise, such as punctuation, stop words, and perform stemming or lemmatization.

2.Feature Engineering:
Create features that capture the semantic similarity between questions. Common features include:
Textual Features: TF-IDF vectors, word embeddings (e.g., Word2Vec, GloVe), and sentence embeddings (e.g., BERT, RoBERTa).
Fuzzy Matching Scores: Levenshtein distance, Jaccard similarity, and cosine similarity.

3.Model Development:
Train machine learning models such as Random Forest or XGBoost.
Evaluate the models using appropriate metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

4.Model Evaluation and Optimization:
Fine-tune hyperparameters and experiment with different model architectures to improve performance.
Perform cross-validation to ensure the model generalizes well to unseen data.

# Challenges

1.Ambiguity in Language: Different phrasings or synonyms can make it difficult for the model to recognize duplicates.

2.Scalability: The system needs to efficiently handle large volumes of data and queries in real-time.

3.Bias and Fairness: Ensuring the model does not exhibit bias towards certain types of questions or user groups.

# Dataset

The most widely known dataset used for the Duplicate Question Pairs problem is the Quora Question Pairs (QQP) dataset. This dataset was released by Quora to help in the development of models that can identify whether two questions have the same intent. The QQP dataset contains pairs of questions with binary labels indicating whether the questions are duplicates (i.e., they have the same meaning) or not.

Structure:

The dataset typically consists of a CSV file with the following columns:

1. id: Unique identifier for the question pair.
2. qid1: Unique identifier for the first question.
3. qid2: Unique identifier for the second question.
4. question1: The text of the first question.
5.question2: The text of the second question.
6. is_duplicate: Binary label indicating if the questions are duplicates (1) or not (0).

 Dataset can be downloaded from the following link-https://www.kaggle.com/c/quora-question-pairs

(IN ORDER TO PERFROM ALL OPERATIONS SMOOTHLY 30000 SAMPLES WERE USED FROM THE TOTAL DATASET OF 404290 SAMPLES)
