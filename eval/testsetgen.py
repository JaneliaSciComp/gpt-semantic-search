
from langchain_community.document_loaders import TextLoader
import os
import pandas as pd

loader = TextLoader("./test.txt")
documents = loader.load()
print(documents)

# %%
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# generator with openai models
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import pandas as pd
import os
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
from langchain_community.embeddings import OllamaEmbeddings

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger('llama_index').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#change the model to a better one once the whole thing is functional
base_url = "http://127.0.0.1:11434/api/generate"
generator_llm = Ollama(model="llama3", base_url=base_url)
critic_llm = Ollama(model="llama3", base_url=base_url)

# generator_llm = Ollama(model="llama3")
# critic_llm = Ollama(model="llama3")

"""curl http://e02u30.int.janelia.org:11434/api/generate -d '{
  "model": "llama3",
  "prompt":"Why is the sky blue?"
}'"""
# generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
# critic_llm = ChatOpenAI(model="gpt-3.5-turbo")

embeddings = OllamaEmbeddings(
    model="llama3"
)
generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
) 


testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
df = testset.to_pandas()
df = df.drop_duplicates(subset='question', keep='first')
questions_list = df['question'].tolist()
seen = set()
questions_list = [x for x in questions_list if not (x in seen or seen.add(x))]

# Assuming df is your DataFrame
df.to_parquet('dataframe.parquet')

"""_______________________"""


import pandas as pd

# Read the Parquet file into df_fix DataFrame
df_fix = pd.read_parquet('df_update.parquet')
columns_to_keep = ['question', 'ground_truth', 'answer', 'contexts']
df_fix= df_fix[columns_to_keep]
df_fix['question'] = df_fix['question'].apply(lambda x: [x] if isinstance(x, str) else x)
df_fix['answer'] = df_fix['answer'].apply(lambda x: [x] if isinstance(x, str) else x)
df_fix['contexts'] = df_fix['contexts'].apply(lambda x: [[y] for y in x] if isinstance(x, list) and all(isinstance(y, str) for y in x) else x)
df_fix['ground_truth'] = df_fix['ground_truth'].apply(lambda x: [[y] for y in x] if isinstance(x, list) and all(isinstance(y, str) for y in x) else x)

from datasets import load_dataset
from datasets import Dataset

dataset_fix = Dataset.from_pandas(df_fix)
dataset_fix = dataset_fix.remove_columns(['__index_level_0__'])

from datasets import Features

def format_columns(example):
    # Format 'question', 'answer', and 'ground_truths' columns to single values
    for column in ['ground_truth', 'answer', 'question']:
        if column in example and example[column]:
            example[column] = example[column]
    
    # Correctly format 'contexts' column to a list of list of strings
    if 'contexts' in example:
        # Ensure 'contexts' is a list of strings (not a list of lists)
        if isinstance(example['contexts'], list):
            # If the items are lists (or other non-string types), flatten and convert to strings
            example['contexts'] = [str(item) for sublist in example['contexts'] for item in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            # If 'contexts' is not a list, convert it into a list of a single string
            example['contexts'] = [str(example['contexts'])]

    
    return example

dataset_fix = dataset_fix.map(format_columns)

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


from ragas import evaluate

eval_llm = Ollama(model="llama3")

result = evaluate(
    dataset_fix,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm=eval_llm,
)

result
result.to_pandas()
result.to_parquet('result.parquet')





