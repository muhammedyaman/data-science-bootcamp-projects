# Text Summarization Agent

AI agent project: summarizing long texts, comparing extractive methods, a local language model and a two-step agent.

## Dataset

`articles/`: 106 Wikipedia articles about human rights (text files from Wikipedia, Wikipedia text is under the CC BY-SA licence). The introduction of each article (the text before the first section heading) is used as the human reference summary; the first 5,000 characters of the rest is the text to summarize. 58 articles pass the length filters, 30 are sampled (seed 42).

## Reference solution

[Text Summarization with Python](https://amanxai.com/2020/12/31/text-summarization-with-python/) (Thecleverprogrammer): word frequencies, sentence score as the sum of word frequencies, the best sentences returned.

Issues found: no evaluation at all; capitalized words are counted with their capitals but looked up in lower case, so they never match; the sum favours long sentences; sentences are returned in the order of their score.

## Our approach

1. Reproduced the reference and compared it with the first 3 sentences, a corrected frequency method and TextRank (TF-IDF cosine similarity, PageRank with networkx).
2. Added a local Llama 3.2 model (Ollama, no API key) and a two-step agent (TextRank selects 8 sentences, the model writes a 3-sentence summary). The model outputs are cached in `llm_summaries.json`.
3. Scored every method with ROUGE-1, ROUGE-2 and ROUGE-L against the human introductions.

## Results (mean F-score, 30 articles)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| First 3 sentences | 0.287 | 0.076 | 0.176 |
| Reference (frequency) | 0.320 | 0.080 | 0.179 |
| Frequency, corrected | 0.269 | 0.087 | 0.169 |
| TextRank | 0.311 | 0.095 | 0.195 |
| Language model | 0.352 | 0.105 | 0.198 |
| Agent (TextRank + language model) | 0.315 | 0.089 | 0.190 |

The language model has the best scores but writes new words (28% of its content words are not in the text). See the notebook's Conclusion.
