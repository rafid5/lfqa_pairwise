### How Well Do LLMs Judge Long-Form Answers? Evaluating LLMs and Fine-Tuned Models on Pairwise Answer Comparisons

### Dependencies
numpy
pandas
scikit-learn
tqdm
python-dotenv
torch
transformers
datasets
sentence-transformers
accelerate
peft
bitsandbytes
rank-bm25
flash-attn
xformers
tokenizers
evaluate
matplotlib
seaborn
openrouter-python
requests
spacy
nltk
language-tool-python
beautifulsoup4
regex
jsonlines
loguru
rich




### How to Run the Training Pipeline


- Download dataset: https://huggingface.co/datasets/nlpatunt/LFQA-HP-1M-Sample

- Adjust path in the python files

- Insert Openrouter and hugginface key in the /config/.env file

- In the training folder, it contains code for trainable model. Each file can be simply run using python <filename.py>

- GPT4o, Llama-4, Gemini-2.5 can be run using python LLM-as-a-judge.py
