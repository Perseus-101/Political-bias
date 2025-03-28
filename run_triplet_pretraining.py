import os
import argparse
import torch
import pandas as pd
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from data_loader import get_dataloaders
from triplet_data_loader import get_triplet_dataloader
from models import TripletTransformer
from triplet_trainer import TripletPreTrainer
from trainer import BiasTrainer
from safetensors.torch import save_file  # Import safetensors for saving

def parse_args():
    parser = argparse.ArgumentParser(description='Triplet Loss Pre-training for Political Bias Detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the dataset')
    parser.add_argument('--split_type', type=str, choices=['media', 'random'], default='random',
                        help='Type of dataset split to use')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenization')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='HuggingFace model name')
    
    # Pre-training arguments
    parser.add_argument('--pretrain_batch_size', type=int, default=8,
                        help='Batch size for pre-training')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='Number of epochs for pre-training')
    parser.add_argument('--pretrain_lr', type=float, default=2e-5,
                        help='Learning rate for pre-training')
    parser.add_argument('--pretrain_grad_accum', type=int, default=4,
                        help='Gradient accumulation steps for pre-training')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune_batch_size', type=int, default=16,
                        help='Batch size for fine-tuning')
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--finetune_lr', type=float, default=5e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--finetune_grad_accum', type=int, default=2,
                        help='Gradient accumulation steps for fine-tuning')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps for scheduler')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--model_save_dir', type=str, default='./roberta-bias-detector',
                        help='Directory to save the fine-tuned model for HuggingFace')
    parser.add_argument('--pretrained_weights_path', type=str, default='./pretrained_weights.safetensors',
                        help='Path to save/load pre-trained weights')
    parser.add_argument('--skip_pretraining', action='store_true',
                        help='Skip pre-training and load weights from pretrained_weights_path')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def format_results(results: Dict[str, float], model_name: str, split_type: str) -> pd.DataFrame:
    """Format results as a DataFrame row."""
    return pd.DataFrame({
        'Model': [f"{model_name}-triplet-pretrained"],
        'Split': [split_type],
        'Macro F1': [f"{results['test_macro_f1']:.2f}"],
        'Acc.': [f"{results['test_accuracy']:.2f}"],
        'MAE': [f"{results['test_mae']:.2f}"],
        'Timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
    })

def save_model_for_huggingface(model, tokenizer, output_dir):
    """
    Save model and tokenizer in a format compatible with HuggingFace Hub,
    using safetensors format for the model weights.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base transformer model from TripletTransformer
    base_model = model.transformer
    
    # Save model configuration with num_labels=3 for the 3 bias classes
    config = base_model.config
    config.num_labels = 3  # Explicitly set for 3 classes: left, center, right
    config.id2label = {0: "left", 1: "center", 2: "right"}
    config.label2id = {"left": 0, "center": 1, "right": 2}
    config.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Create a sequence classification model with our base model
    from transformers import AutoModelForSequenceClassification
    
    # First create the model with just the configuration
    sequence_model = AutoModelForSequenceClassification.from_pretrained(
        base_model.config._name_or_path,
        num_labels=3,
        config=config
    )
    
    # Then load the state dictionary from the base model
    # Get the base model state dict
    base_state_dict = base_model.state_dict()
    
    # Load the base model weights that match the sequence model
    sequence_model_dict = sequence_model.state_dict()
    for key in sequence_model_dict.keys():
        if key in base_state_dict and 'classifier' not in key:
            sequence_model_dict[key] = base_state_dict[key]
    
    # Load the updated state dict
    sequence_model.load_state_dict(sequence_model_dict, strict=False)
    
    # Copy the classifier weights from our trained model to the sequence classification model
    sequence_model.classifier.load_state_dict(model.classifier.state_dict())
    
    # Get state dict of the complete model with classifier
    state_dict = sequence_model.state_dict()
    
    # Save model weights using safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)
    
    print(f"Model saved to {output_dir} in safetensors format with 3-class classifier")
    
    # Optionally save a README.md with model information
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# RoBERTa Political Bias Detector\n\n")
        f.write("This model was trained using triplet loss pre-training followed by fine-tuning for political bias detection.\n\n")
        f.write("## Model Details\n\n")
        f.write(f"- Base model: {base_model.config._name_or_path}\n")
        f.write("- Task: Political Bias Detection (3 classes: left, center, right)\n")
        f.write("- Classes: 0=left, 1=center, 2=right\n")

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Phase 1: Pre-training with Triplet Loss
    if not args.skip_pretraining:
        print("\n=== Phase 1: Pre-training with Triplet Loss ===")
        
        # Create triplet dataloader
        print(f"Loading triplet data from {args.split_type} split...")
        triplet_loader = get_triplet_dataloader(
            data_dir=args.data_dir,
            split_type=args.split_type,
            tokenizer_name=args.model_name,
            batch_size=args.pretrain_batch_size,
            max_length=args.max_length
        )
        
        # Create model for pre-training
        print(f"Creating TripletTransformer model with {args.model_name}...")
        pretrain_model = TripletTransformer(args.model_name, num_bias_classes=3)
        
        # Create pre-trainer
        pretrain_trainer = TripletPreTrainer(
            model=pretrain_model,
            triplet_loader=triplet_loader,
            device=device,
            learning_rate=args.pretrain_lr,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.pretrain_grad_accum
        )
        
        # Pre-train model
        print(f"Pre-training for {args.pretrain_epochs} epochs...")
        pretrain_metrics = pretrain_trainer.train(args.pretrain_epochs)
        
        # Save pre-trained weights in safetensors format
        if hasattr(pretrain_trainer, 'save_pretrained_model'):
            # If the trainer has this method, we should modify it to use safetensors
            # For now, we'll call it and add a note about modification
            print("Saving pre-trained weights - Note: This will use PyTorch format, consider updating TripletPreTrainer to use safetensors")
            pretrain_trainer.save_pretrained_model(args.pretrained_weights_path)
        else:
            # Fallback if save_pretrained_model doesn't exist
            print("Saving pre-trained weights")
            state_dict = pretrain_model.state_dict()
            save_file(state_dict, args.pretrained_weights_path)
    
    # Phase 2: Fine-tuning for Political Bias Classification
    print("\n=== Phase 2: Fine-tuning for Political Bias Classification ===")
    
    # Get dataloaders for fine-tuning
    print(f"Loading {args.split_type} split data for fine-tuning...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        split_type=args.split_type,
        tokenizer_name=args.model_name,
        batch_size=args.finetune_batch_size,
        max_length=args.max_length
    )
    
    # Create model for fine-tuning
    print(f"Creating TripletTransformer model for fine-tuning...")
    finetune_model = TripletTransformer(args.model_name, num_bias_classes=3)
    
    # Load pre-trained weights if available
    if os.path.exists(args.pretrained_weights_path):
        print(f"Loading pre-trained weights from {args.pretrained_weights_path}")
        
        try:
            # First try loading with safetensors
            from safetensors.torch import load_file
            state_dict = load_file(args.pretrained_weights_path)
            finetune_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Safetensors loading failed: {e}")
            print("Trying to load with PyTorch format instead...")
            # Fall back to PyTorch loading
            state_dict = torch.load(args.pretrained_weights_path)
            finetune_model.load_state_dict(state_dict)
    else:
        print("No pre-trained weights found. Starting fine-tuning from scratch.")
    
    # Create trainer for fine-tuning
    finetune_trainer = BiasTrainer(
        model=finetune_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.finetune_lr,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.finetune_grad_accum
    )
    
    # Fine-tune model
    print(f"Fine-tuning for {args.finetune_epochs} epochs...")
    finetune_metrics = finetune_trainer.train(args.finetune_epochs)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = finetune_trainer.evaluate('test')
    print(f"Test metrics: {' '.join(f'{k}: {v:.2f}' for k, v in test_metrics.items())}")
    
    # Format results
    results_df = format_results(test_metrics, args.model_name, args.split_type)
    
    # Save to consolidated results file
    consolidated_results_path = os.path.join(args.output_dir, "consolidated_results.csv")
    
    # Check if consolidated results file exists and append to it
    if os.path.exists(consolidated_results_path):
        existing_results = pd.read_csv(consolidated_results_path)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
        updated_results.to_csv(consolidated_results_path, index=False)
    else:
        # Create new file if it doesn't exist
        results_df.to_csv(consolidated_results_path, index=False)
    
    print(f"Results saved to {consolidated_results_path}")
    
    # Print results table
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    # Save model in HuggingFace format using safetensors
    print("\n=== Saving Model for HuggingFace Hub ===")
    save_model_for_huggingface(finetune_model, tokenizer, args.model_save_dir)

if __name__ == "__main__":
    main()