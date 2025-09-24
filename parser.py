# parser.py
import argparse

def get_parser():
    """
    Creates an argument parser for training or evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Argument parser for Swin Regression model training."
    )

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset folder.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Path to the dataset folder.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation.")

    # Training arguments
    parser.add_argument("--hyperparam_ft",type=int, default=0, 
                        help="perform hyperparameter finetuning" )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for the optimizer.")

    # Miscellaneous
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--save_model", type=int, default=1, help="save the trained model")
    parser.add_argument("--load_model", type=int,default=0, help="load the model saved")
    
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
