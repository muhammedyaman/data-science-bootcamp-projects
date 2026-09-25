# Resume Screening Agent

AI agent project: rank resumes for a job description with a search tool, and score them with a local language model.

## Dataset

`UpdatedResumeDataSet.csv` (962 rows, 25 categories), the dataset used by the reference article (originally a Kaggle resume dataset). Only 166 rows are different resumes.

## Reference solution

[Resume Screening with Python](https://amanxai.com/2020/12/06/resume-screening-with-python/) (Aman Kharwal): TF-IDF with 1,500 features and a One-vs-Rest KNN classifier, reported accuracy 0.99.

Issues found: 796 of the 962 rows are exact copies, and 99.5% of the test resumes have a copy in the training set, so the accuracy measures memory; the cleaning function deletes "cc" inside words.

## Our approach

1. Reproduced the reference, then evaluated KNN, Logistic Regression and Naive Bayes on the different resumes only.
2. Built a workflow agent: a TF-IDF search tool that ranks resumes for a job description (25 job descriptions written for the notebook), and a scoring tool with a local Llama 3.2 model (Ollama, no API key, scores cached in `llm_scores.json`).
3. Measured the search tool (precision at 5, mean reciprocal rank) and whether the model score improves the ranking (AUC).

## Results

| Model / tool | Result |
|---|---|
| KNN (reference), copies in the split | accuracy 0.99 |
| KNN, different resumes only | accuracy 0.735 |
| Logistic Regression, different resumes only | accuracy 0.324 |
| Naive Bayes, different resumes only | accuracy 0.118 |
| TF-IDF search tool | precision at 5 = 0.792 (random 0.016) |
| Search + LLM score (12 jobs) | AUC 0.715 against 0.800 for the search similarity alone |

See the notebook's Conclusion for limitations.
