import os
import argparse
from llm_bias_detector import LLMBiasDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM Political Bias Benchmark')
    
    # LLM arguments
    parser.add_argument('--models', type=str, nargs='+',
                        default=['meta-llama/Llama-3-8b-hf', 'meta-llama/Llama-3-70b-hf',
                                 'mistralai/Mistral-7B-Instruct-v0.2', 'tiiuae/falcon-7b'],
                        help='List of LLM models to benchmark')
    
    # Bias detection arguments
    parser.add_argument('--bias_model_name', type=str, default='roberta-base',
                        help='Name of the pre-trained model used for bias detection')
    parser.add_argument('--bias_model_path', type=str, default='./results/roberta-base-bias-model.pt',
                        help='Path to the fine-tuned bias detection model weights')
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='./llm_benchmark_results',
                        help='Directory to save benchmark results')
    parser.add_argument('--topics_file', type=str, default='./topics.json',
                        help='Path to JSON file containing topics')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to run models on')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize LLM Bias Detector with the first model (will be replaced during benchmarking)
    detector = LLMBiasDetector(
        llm_model_name=args.models[0],
        bias_model_path=args.bias_model_path,
        bias_model_name=args.bias_model_name,
        device=args.device
    )
    
    # Run benchmark
    results = detector.benchmark_models(
        model_names=args.models,
        topics=None,  # Will load from topics_file in the LLMBiasDetector
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n=== Benchmark Summary ===\n")
    model_bias_summary = results.groupby('model')['bias_label'].value_counts().unstack().fillna(0)
    print(model_bias_summary)
    
    # Calculate bias scores (higher means more right-leaning)
    bias_scores = {}
    for model in results['model'].unique():
        model_results = results[results['model'] == model]
        left_count = sum(model_results['bias_label'] == 'left')
        center_count = sum(model_results['bias_label'] == 'center')
        right_count = sum(model_results['bias_label'] == 'right')
        total = left_count + center_count + right_count
        
        # Calculate weighted score: -1 for left, 0 for center, 1 for right
        bias_score = (right_count - left_count) / total
        bias_scores[model] = bias_score
    
    print("\nBias Scores (negative = left-leaning, positive = right-leaning):")
    for model, score in bias_scores.items():
        print(f"{model}: {score:.2f}")

if __name__ == "__main__":
    main()