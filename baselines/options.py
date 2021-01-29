import os
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default="../annotations.json")
    parser.add_argument("--imagepath", type=str, default="../images")
    parser.add_argument("--exp_id", type=str)
    parser.add_argument(
        "--log_dir", type=str, default=os.path.join("./runs", str(datetime.now())[:-7])
    )
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--resume", type=str, help="model checkpoint to resume")
    parser.add_argument(
        "--train_split", type=str, default="train", choices=["train", "train_valid"]
    )
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--visual_feature_size", type=int, default=3)
    parser.add_argument("--predicate_embedding_dim", type=int, default=512)

    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=512)

    parser.add_argument(
        "--model",
        type=str,
        choices=["drnet", "vtranse", "vipcnn", "pprfcn"],
        default="drnet",
    )

    # VipCNN
    parser.add_argument("--roi_size", type=int, default=6)
    # VipCNN & PPR-FCN
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet34", "resnet101"],
        default="resnet18",
    )

    args = parser.parse_args()
    args.predicate_categories = [
        "above",
        "behind",
        "in",
        "in front of",
        "next to",
        "on",
        "to the left of",
        "to the right of",
        "under",
    ]
    args.max_phrase_len = 2

    if args.exp_id != None:
        args.log_dir = os.path.join("./runs", args.exp_id)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        os.makedirs(os.path.join(args.log_dir, "predictions"))
        os.makedirs(os.path.join(args.log_dir, "checkpoints"))

    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
