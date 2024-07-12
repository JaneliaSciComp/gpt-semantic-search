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
from langchain_community.llms import Ollama

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



