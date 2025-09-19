import os
import argparse
import yaml
import json
import torch
import numpy as np
import random
import logging
from . import preprocess, train, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise

def run_experiment(config):
    """Runs a single experiment for one seed."""
    set_seed(config['training']['seed'])
    logger.info(f"Running experiment for seed: {config['training']['seed']}")

    model, tokenizer = train.setup_model_and_tokenizer(config)
    datasets = preprocess.load_and_preprocess_data(config, tokenizer)
    
    trainer, train_result = train.train_model(
        config, 
        model, 
        tokenizer, 
        datasets, 
        evaluate.compute_sst2_metrics
    )

    results = evaluate.evaluate_model_performance(trainer, tokenizer, config, train_result)
    
    # Save final model
    output_dir = os.path.join(config['output_dir'], f"seed_{config['training']['seed']}", "final_model")
    trainer.save_model(output_dir)
    logger.info(f"Final model saved to {output_dir}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run ProA-JSP experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run a small-scale smoke test.")
    group.add_argument("--full-experiment", action="store_true", help="Run the full experiment.")
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    config = load_config(config_path)

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This experiment requires a GPU.")
        return

    os.makedirs(config['output_dir'], exist_ok=True)

    if args.smoke_test:
        config['training']['seed'] = config['training'].get('seed', 42) # Ensure seed is set
        run_experiment(config)
    elif args.full_experiment:
        all_results = []
        seeds = config['training'].get('seeds', [12, 23, 34])
        for seed in seeds:
            run_config = config.copy()
            run_config['training']['seed'] = seed
            try:
                result = run_experiment(run_config)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Experiment failed for seed {seed}: {e}", exc_info=True)
        
        # Aggregate results
        if all_results:
            aggregated_results = {}
            keys = all_results[0].keys()
            for key in keys:
                if isinstance(all_results[0][key], (int, float)):
                    values = [r[key] for r in all_results if r.get(key) is not None]
                    if values:
                        aggregated_results[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values)
                        }
            
            # Save and print aggregated results
            summary_path = os.path.join(config['output_dir'], 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(aggregated_results, f, indent=4)
            logger.info(f"Aggregated results saved to {summary_path}")
            print("\n--- AGGREGATED RESULTS ---")
            print(json.dumps(aggregated_results, indent=4))

if __name__ == "__main__":
    main()
