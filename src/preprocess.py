from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorWithPadding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(config, tokenizer):
    """Loads and preprocesses the main task dataset and anchor pool."""
    try:
        # Load main task dataset
        logger.info(f"Loading dataset: {config['task']['name']}/{config['task']['subset']}")
        raw_datasets = load_dataset(config['task']['name'], config['task']['subset'])

        # SST-2 validation set is small, use it as is. Split train for validation if needed.
        if 'validation' not in raw_datasets:
             logger.warning("No validation split found. Splitting train set 90/10.")
             split_dataset = raw_datasets['train'].train_test_split(test_size=0.1, seed=config['training']['seed'])
             raw_datasets['train'] = split_dataset['train']
             raw_datasets['validation'] = split_dataset['test']

        text_column = config['task']['text_column']
        label_column = config['task']['label_column']
        max_length = config['task']['max_length']

        def preprocess_function(examples):
            # Tokenize the text
            result = tokenizer(examples[text_column], truncation=True, max_length=max_length)
            result["labels"] = examples[label_column]
            return result

        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )

        # Load anchor pool for ProA-JSP
        anchor_dataset = load_anchor_pool(tokenizer, max_length, seed=config['training']['seed'])

        return {
            'train': tokenized_datasets['train'],
            'validation': tokenized_datasets['validation'],
            'anchor': anchor_dataset,
            'collator': DataCollatorWithPadding(tokenizer=tokenizer)
        }
    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {e}")
        raise

def load_anchor_pool(tokenizer, max_length, seed, num_samples=50000):
    """Loads and preprocesses the anchor pool from C4 and other critical sources."""
    try:
        logger.info("Loading anchor pool...")
        # General-domain sentences (C4)
        c4_dataset = load_dataset('c4', 'en', split=f'train[:{num_samples}]', streaming=True).shuffle(seed=seed).take(num_samples)
        c4_df = list(c4_dataset)
        from datasets import Dataset
        c4_dataset = Dataset.from_list(c4_df)
        
        # For simplicity, we only use C4 here. A full implementation would also load
        # and combine Wiki-verified and TruthfulQA statement datasets.

        def anchor_preprocess(examples):
            return tokenizer(examples['text'], truncation=True, max_length=max_length)

        tokenized_anchor = c4_dataset.map(
            anchor_preprocess, 
            batched=True, 
            remove_columns=c4_dataset.column_names
        )
        logger.info(f"Anchor pool loaded with {len(tokenized_anchor)} samples.")
        return tokenized_anchor
    except Exception as e:
        logger.error(f"Could not load anchor pool: {e}")
        # Return a dummy dataset to prevent crashing if anchor pool fails, with a warning.
        logger.warning("Failed to load anchor pool. ProA-JSP might not function correctly.")
        return None
