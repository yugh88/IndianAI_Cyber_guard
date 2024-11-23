CyberGuard AI Hackathon - NLP Complaint Categorization Model

Overview

This project was developed for the IndiaAI CyberGuard AI Hackathon, aiming to build an NLP model that classifies complaints based on categories such as Online Financial Fraud, Cyber Attacks, and subcategories like Fraud Callvishing, SQL Injection, etc. The model leverages machine learning and deep learning techniques to handle large, imbalanced text datasets and classify complaints efficiently.

Team

	•	Yugh Juneja
	•	Mukul Gupta
	•	Unnati Sabu

Problem Statement

The primary objective is to categorize complaints based on victim type, fraud type, and other relevant parameters. The dataset contains text data with 15 primary categories and 35 subcategories. The challenge lies in ensuring consistent classification across both the training and test datasets while handling imbalanced and complex data.

Approach

Data Preprocessing

	1.	Text Cleaning: Removal of non-alphanumeric words and null values.
	2.	Tokenization: Breaking down text into meaningful units (tokens).
	3.	Lemmatization: Reducing words to their base form.
	4.	Visualization: Used countplot, stacked bar chart, and word cloud for data exploration and insight.

Models Used

	•	Random Forest: Used to handle imbalanced data and high-dimensional text data.
	•	Accuracy for primary categories: 98%
	•	Accuracy for subcategories: 52%
	•	Optimized using RandomizedSearchCV for hyperparameter tuning.
	•	Recurrent Neural Networks (RNN): Applied for sequential pattern recognition in text.
	•	Training with 50 epochs, Relu activation, and Adam optimizer.
	•	Accuracy results for subcategories were not included but expected improvements due to sequential learning.

Hyperparameter Tuning (Random Forest)

param_grid = {
  'n_estimators': [100, 300, 500],
  'max_depth': [10, 20, 30],
  'max_features': ['sqrt', 'log2', 0.5],
  'class_weight': ['balanced'],
  'bootstrap': [True],
}

Results

	•	Random Forest:
	•	Primary categories: 98% accuracy.
	•	Subcategories: 52% accuracy.
	•	RNN:
	•	Subcategory classification still needs improvement, but RNN’s ability to capture sequential patterns has shown promise.

Future Work

	•	Advanced Models: Incorporate BERT for better handling of large text datasets with deep contextual relationships.
	•	Ensemble Techniques: Explore methods like XGBoost and LightGBM for improved results on larger and imbalanced datasets.
	•	Additional Optimizations: Further hyperparameter tuning and model refinement for both Random Forest and RNN.

Installation

	1.	Clone this repository:

git clone https://github.com/yourusername/cyberguard-ai-hackathon.git


	2.	Install required libraries:

pip install -r requirements.txt


	3.	Ensure that the necessary datasets (train and test) are in the data/ directory.

How to Run

	1.	Preprocessing: Run the preprocessing script to clean and tokenize the data:

python preprocessing.py


	2.	Model Training: To train the Random Forest or RNN model, run:

python train_model.py


	3.	Evaluation: After training, evaluate the model’s performance on the test dataset:

python evaluate_model.py



Results and Metrics

	•	Random Forest:
	•	Accuracy (Categories): 98%
	•	Accuracy (Subcategories): 52%
	•	RNN: Performance yet to be fully detailed, but expected improvements in subcategory classification.

Conclusion

The project presents an effective NLP approach for automating text classification in cybersecurity complaints. The combination of Random Forest and RNN models helps balance efficiency with the complexity of handling imbalanced datasets. Further optimizations, including the use of advanced deep learning models like BERT, are expected to enhance classification accuracy, particularly for subcategories.

References

	•	Text Classification with RNN (TensorFlow)
	•	Article on Deep Learning Models for Text Classification

License

This project is licensed under the MIT License - see the LICENSE file for details.
