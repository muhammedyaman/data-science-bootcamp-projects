# Chatbot Agent

AI agent project: a chatbot that classifies the intent of a message, answers with a prepared reply when it is confident, and hands other messages to a local language model.

## Dataset

`intents.json`: the intent file of the reference article (8 intents, 33 example messages). The test messages (80 in-scope paraphrases and 50 out-of-scope messages) are written in the notebook.

## Reference solution

[Create a Chatbot with Python and Machine Learning](https://amanxai.com/2020/11/01/chatbot-with-machine-learning-and-python/) (Aman Kharwal): embedding network trained for 500 epochs on the example messages, random prepared reply of the predicted intent.

Issues found: no test on new messages, the network memorizes the training messages, and the bot always answers, even to unrelated messages.

## Our approach

1. Tested the reference network on new messages and compared it with TF-IDF + Logistic Regression and a nearest-message model.
2. Chose a confidence threshold on a development half of the test messages, so the bot can decide that a message is out of scope.
3. Built a workflow agent: classifier, then prepared reply or local Llama 3.2 (run by Ollama, no API key). A fixed workflow was chosen instead of a complex agent framework.

## Results (test half)

| Model | In-scope accuracy | Out-of-scope rejected (with threshold) |
|---|---|---|
| Reference network | 0.550 | 0.92 (accepts only 15% of in-scope) |
| TF-IDF + Logistic Regression | 0.750 | 0.80 |
| Nearest training message | 0.725 | 0.84 |

The reference network scores 1.0 on its training messages. The test set is small; see the notebook's Conclusion.
