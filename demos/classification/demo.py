import sys
import os
import pickle
from typing import Any
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from promptwizard.glue.promptopt.instantiate import GluePromptOpt
from promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

# Load environment variables from .env file
load_dotenv(override=True)

class Classification(DatasetSpecificProcessing):
    """
    A custom class for the classification task, inheriting from DatasetSpecificProcessing.
    This class handles dataset-specific processing and evaluation logic.
    """
    def dataset_to_jsonl(self, dataset_jsonl: str, **kwargs: Any) -> None:
        """
        This function is not needed as we are providing the dataset in jsonl format directly.
        """
        pass

    def extract_final_answer(self, answer: str):
        """
        Extracts the final answer from the model's output.
        For classification, the output is expected to be one of the labels.
        """
        return answer.strip()

    def access_answer(self, llm_output: str, gt_answer: str):
        """
        Compares the model's output with the ground truth answer.
        """
        predicted_answer = self.extract_final_answer(llm_output)
        is_correct = False
        if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
            is_correct = True
        return is_correct, predicted_answer

def main():
    """
    Main function to run the prompt optimization and evaluation.
    """
    # Initialize the custom processor
    classification_processor = Classification()

    # Define file and directory paths
    # Note: This script assumes it is run from the 'demos/classification' directory.
    # If run from the root, the paths might need adjustment.
    current_dir = os.path.dirname(__file__)
    train_file_name = os.path.join(current_dir, "data", "train.jsonl")
    test_file_name = os.path.join(current_dir, "data", "test.jsonl")
    path_to_config = os.path.join(current_dir, "configs")
    promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

    # Create an object for prompt optimization
    gp = GluePromptOpt(promptopt_config_path,
                       setup_config_path,
                       train_file_name,
                       classification_processor)

    # Get the best prompt
    print("Starting prompt optimization...")
    best_prompt, expert_profile = gp.get_best_prompt(use_examples=False, run_without_train_examples=False, generate_synthetic_examples=False)
    
    # Save and print the best prompt and expert profile
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, "best_prompt.pkl"), 'wb') as f:
        pickle.dump(best_prompt, f)

    with open(os.path.join(results_dir, "expert_profile.pkl"), 'wb') as f:
        pickle.dump(expert_profile, f)

    print(f"Best prompt found: {best_prompt}")
    print(f"Expert profile: {expert_profile}")

    # Evaluate the best prompt on the test set
    print("\nEvaluating the best prompt on the test set...")
    gp.EXPERT_PROFILE = expert_profile
    gp.BEST_PROMPT = best_prompt
    accuracy = gp.evaluate(test_file_name)

    print(f"\nFinal Accuracy on the test set: {accuracy}")

if __name__ == "__main__":
    main()