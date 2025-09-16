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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader.")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use for training.")
    # Miscellaneous
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints.")

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
