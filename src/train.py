import os
import copy
import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(config):
    """Loads the model and tokenizer based on the provided configuration."""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=config['model'].get('load_in_8bit', False),
            load_in_4bit=config['model'].get('load_in_4bit', False),
        )

        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            quantization_config=bnb_config if bnb_config.load_in_4bit or bnb_config.load_in_8bit else None,
            device_map=config['model']['device_map'],
            token=hf_token,
            torch_dtype=torch.bfloat16 if config['training'].get('bf16') else torch.float32,
        )

        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        if bnb_config.load_in_8bit or bnb_config.load_in_4bit:
             model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error setting up model and tokenizer: {e}")
        raise

def jensen_shannon_divergence(p, q, reduction='mean'):
    """Computes Jensen-Shannon Divergence between two logit distributions."""
    p_probs = F.softmax(p, dim=-1)
    q_probs = F.softmax(q, dim=-1)
    m_probs = 0.5 * (p_probs + q_probs)
    
    # Use log_softmax for numerical stability
    kl_p_m = F.kl_div(F.log_softmax(p, dim=-1), m_probs, reduction=reduction)
    kl_q_m = F.kl_div(F.log_softmax(q, dim=-1), m_probs, reduction=reduction)
    
    return 0.5 * (kl_p_m + kl_q_m)

class ProAJSPTrainer(Trainer):
    """
    Custom Trainer implementing the ProA-JSP loss.
    """
    def __init__(self, *args, teacher_model=None, proa_jsp_config=None, anchor_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.proa_jsp_config = proa_jsp_config
        if self.proa_jsp_config.get('enabled', False):
            if teacher_model is None or anchor_dataset is None:
                raise ValueError("Teacher model and anchor dataset are required for ProA-JSP.")
            self.teacher_model = teacher_model.to(self.args.device)
            self.teacher_model.eval()
            self.anchor_dataset = anchor_dataset
            self.anchor_loader = self.get_train_dataloader(anchor_dataset)
            self.anchor_iterator = iter(self.anchor_loader)
            logger.info("ProA-JSP is enabled.")

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to add the ProA-JSP regularization term."""
        # Base cross-entropy loss from the task
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if not self.proa_jsp_config.get('enabled', False) or not model.training:
            return (base_loss, outputs) if return_outputs else base_loss

        try:
            anchor_batch = next(self.anchor_iterator)
        except StopIteration:
            self.anchor_iterator = iter(self.anchor_loader)
            anchor_batch = next(self.anchor_iterator)

        # Move anchor batch to the correct device
        anchor_inputs = {k: v.to(self.args.device) for k, v in anchor_batch.items() if isinstance(v, torch.Tensor)}
        anchor_input_ids = anchor_inputs.get('input_ids')
        if anchor_input_ids is None:
            logger.warning("Could not find 'input_ids' in anchor batch. Skipping JSP loss.")
            return (base_loss, outputs) if return_outputs else base_loss

        # Get logits from student and teacher
        with torch.no_grad():
            teacher_logits = self.teacher_model(anchor_input_ids).logits
        student_logits = model(anchor_input_ids).logits

        # Align dimensions if necessary (e.g., sequence length)
        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]

        # Compute JSD loss
        # NOTE: Saliency-guided importance weights and dual-teacher trust region are simplified here.
        # A full implementation would use Captum for Integrated Gradients and manage multiple teacher snapshots.
        jsp_loss = jensen_shannon_divergence(student_logits, teacher_logits)
        
        lambda_jsp = self.proa_jsp_config.get('lambda', 1.0)
        total_loss = base_loss + lambda_jsp * jsp_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def get_train_dataloader(self, train_dataset):
        """Helper to create a dataloader for the anchor dataset."""
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def train_model(config, model, tokenizer, datasets, compute_metrics_fn):
    """Main function to set up and run the training process."""
    
    # Create a frozen copy of the initial model to act as the teacher (theta_0)
    teacher_model = copy.deepcopy(model)
    for param in teacher_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=os.path.join(config['output_dir'], f"seed_{config['training']['seed']}"),
        num_train_epochs=config['training']['epochs'],
        max_steps=config['training'].get('max_steps', -1),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        bf16=config['training'].get('bf16', False),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=config['training']['seed']
    )

    trainer = ProAJSPTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer,
        data_collator=datasets['collator'],
        compute_metrics=compute_metrics_fn,
        teacher_model=teacher_model,
        proa_jsp_config=config.get('proa_jsp', {}),
        anchor_dataset=datasets.get('anchor')
    )

    logger.info(f"Starting training for seed {config['training']['seed']}...")
    train_result = trainer.train()
    logger.info("Training finished.")

    return trainer, train_result
