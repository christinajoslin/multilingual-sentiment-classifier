# Multilingual-Sentiment-Classifier

This repository contains an end-to-end Natural Language Processing (NLP) pipeline for multilingual sentiment analysis using the pre-trained BERT model `nlptown/bert-base-multilingual-uncased-sentiment`. The pipeline includes data preprocessing, model fine-tuning with Hugging Face's Trainer, and evaluation metrics. The fine-tuned model classifies text sentiment as negative, neutral, or positive, achieving a test accuracy of 77% and a test F1 score of 76%. Due to limited computational resources, the training process used only 84,000 examples (out of 1.26 million available) and took approximately 1 hour on a T4 GPU.

## Features
- **Data Preprocessing:** Filters datasets to target languages (English, Spanish, French, German), converts star ratings to sentiment polarity, and performs stratified sampling to maintain class balance.
- **Tokenization:** Prepares text data for the BERT model by applying padding, truncation, and a maximum length of 128 tokens.
- **Model Fine-Tuning:** Utilizes `nlptown/bert-base-multilingual-uncased-sentiment` for sequence classification with custom training configurations.
- **Evaluation Metrics:** Reports accuracy, precision, recall, and F1-score on training, development, and test sets.
- **Custom Callbacks:** Includes early stopping for optimal performance.
- **Final Predictions:** Outputs predictions for test data in a structured CSV format.

## Why `nlptown/bert-base-multilingual-uncased-sentiment?`
This model is specifically designed for multilingual sentiment analysis and offers the following advantages:
- **Multilingual Support:** Supports multiple languages, including English, Spanish, French, and German, making it well-suited for diverse datasets like the Amazon Reviews Corpus.
- **Pre-Trained for Sentiment Analysis:** Fine-tuned for sentiment classification tasks, reducing the need for extensive retraining and ensuring reliable performance out of the box.
- **Robust Contextual Understanding:** Leverages the BERT architecture for better handling of complex sentence structures and nuanced sentiments across different languages.

## Instructions
1. Ensure you have the required libraries installed (`transformers`, `datasets`, `pandas`, `numpy`, `scikit-learn`).
2. Place the training, validation, and test datasets (`train.csv`, `validation.csv`, `test.csv`) in the appropriate directory.
3. Run the script sequentially to preprocess data, fine-tune the model, and evaluate its performance.
4. The final predictions will be saved in a CSV file in the specified output directory.

## Dependencies
- `transformers`: For tokenization and fine-tuning the BERT model.
- `datasets`: For handling datasets compatible with Hugging Face's Trainer.
- `pandas`: For data manipulation and processing.
- `numpy`: For numerical computations.
- `scikit-learn`: For evaluation metrics such as accuracy, precision, recall, and F1-score.

## Notes
- Ensure the file paths in the script match the locations of your datasets.
- Training arguments can be adjusted to explore different configurations (e.g., learning rate, batch size, epochs).
- The `random_state = 42` is used throughout to ensure reproducibility.

## Author
Christina Joslin  

## Acknowledgements
- Data provided by the [Multilingual Amazon Reviews Corpus posted on Kaggle](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi).
- **Citation:** Phillip Keung, Yichao Lu, György Szarvas, and Noah A. Smith. “The Multilingual Amazon Reviews Corpus.” In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020.https://www.amazon.science/publications/the-multilingual-amazon-reviews-corpus
- Special thanks to the open-source community for contributing the libraries and tools used in this project.

