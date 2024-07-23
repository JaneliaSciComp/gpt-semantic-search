import pandas as pd
from datasets import Dataset
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
import logging
from tqdm import tqdm
from ragas.run_config import RunConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

df_fix = pd.read_parquet('withAnswersTestset.parquet')

eval_llm = Ollama(model="llama3", timeout=300) 
eval_embeddings = OllamaEmbeddings(model="avr/sfr-embedding-mistral")

metrics = [context_precision, faithfulness, answer_relevancy, context_recall]

batch_size = 5
results = []

for i in tqdm(range(0, len(df_fix), batch_size)):
    batch_df = df_fix.iloc[i:i+batch_size]
    try:
        batch_dataset = Dataset.from_pandas(batch_df)
        batch_dataset = batch_dataset.map(format_columns)

        run_config = RunConfig(timeout= 360, max_retries=2, max_wait=60, max_workers=32, log_tenacity=True)
        
        batch_result = evaluate(
            batch_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
        results.append(batch_result)
    except Exception as e:
        logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
        continue

if results:
    final_result = pd.concat([r.to_pandas() for r in results], ignore_index=True)
    final_result.to_parquet('results.parquet')
    logging.info(f"Evaluation completed. Results saved to 'results.parquet'")
    logging.info(f"Final shape of results: {final_result.shape}")
else:
    logging.warning("No results were successfully processed.")