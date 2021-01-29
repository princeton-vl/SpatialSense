import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torchvision.models as models
import numpy as np

np.random.seed(0)
import random
import pickle
import argparse
import json
from progressbar import ProgressBar
from collections import defaultdict
from spatial_features import raw_spatial_feature
from appearance_features import raw_appearance_feature
from masked_regression import MaskedRegression


predicate_categories = [
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


def load_features(opts):
    print("Loading features..")
    gmm = pickle.load(open(opts.gmm_path, "rb"))
    pca = pickle.load(open(opts.pca_path, "rb"))
    data = defaultdict(
        lambda: {
            "raw_spatial_feature": [],
            "discretized_spatial_feature": [],
            "appearance_feature": [],
            "predicate": [],
            "label": [],
            "_id": [],
        }
    )
    dataset = json.load(open(opts.data_path))
    bar = ProgressBar(max_value=len(dataset))
    for i, img in enumerate(dataset):
        for annot in img["annotations"]:
            raw_fea = raw_spatial_feature(
                annot["subject"]["bbox"], annot["object"]["bbox"]
            ).astype(np.float32)
            discretized_fea = (
                gmm.predict_proba(raw_fea.reshape(1, -1)).squeeze().astype(np.float32)
            )
            predicate_onehot = np.zeros(len(predicate_categories), dtype=np.float32)
            predicate_onehot[predicate_categories.index(annot["predicate"])] = 1.0
            data[img["split"]]["raw_spatial_feature"].append(raw_fea)
            data[img["split"]]["discretized_spatial_feature"].append(discretized_fea)
            data[img["split"]]["predicate"].append(predicate_onehot)
            data[img["split"]]["label"].append(annot["label"])
            data[img["split"]]["_id"].append(annot["_id"])
            if opts.appr:
                appr_fea = pca.transform(
                    np.vstack(
                        raw_appearance_feature(
                            img["url"],
                            annot["subject"]["bbox"],
                            annot["object"]["bbox"],
                        )
                    )
                )
                appr_fea = np.hstack(appr_fea)
                appr_fea /= np.linalg.norm(appr_fea)
                data[img["split"]]["appearance_feature"].append(appr_fea)
        bar.update(i)

    if opts.no_val:
        data["train"]["raw_spatial_feature"].extend(
            data["valid"]["raw_spatial_feature"]
        )
        data["train"]["discretized_spatial_feature"].extend(
            data["valid"]["discretized_spatial_feature"]
        )
        data["train"]["predicate"].extend(data["valid"]["predicate"])
        data["train"]["label"].extend(data["valid"]["label"])
        data["train"]["_id"].extend(data["valid"]["_id"])
        if opts.appr:
            data["train"]["appearance_feature"].extend(
                data["valid"]["appearance_feature"]
            )

    for split in data.keys():
        indexes = list(range(len(data[split]["raw_spatial_feature"])))
        random.shuffle(indexes)
        print("%d samples in %s" % (len(indexes), split))
        data[split]["raw_spatial_feature"] = torch.Tensor(
            np.vstack(data[split]["raw_spatial_feature"])[indexes]
        ).cuda()
        data[split]["discretized_spatial_feature"] = torch.Tensor(
            np.vstack(data[split]["discretized_spatial_feature"])[indexes]
        ).cuda()
        data[split]["predicate"] = (
            torch.Tensor(np.vstack(data[split]["predicate"])[indexes]).byte().cuda()
        )
        data[split]["label"] = torch.Tensor(
            [1.0 if label == True else -1.0 for label in data[split]["label"]]
        )[indexes].cuda()
        data[split]["_id"] = list(np.asarray(data[split]["_id"])[indexes])
        if opts.appr:
            data[split]["appearance_feature"] = torch.Tensor(
                np.vstack(data[split]["appearance_feature"])[indexes]
            ).cuda()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../annotations.json")
    parser.add_argument("--gmm_path", type=str, default="gmm.pickle")
    parser.add_argument("--pca_path", type=str, default="pca.pickle")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--appr", action="store_true")
    parser.add_argument("--no_val", action="store_true")
    opts = parser.parse_args()
    if opts.appr:
        opts.l2 = 1e-4
    else:
        opts.l2 = 1e-6
    print(opts)

    data = load_features(opts)
    model = MaskedRegression(opts)
    model.cuda()
    criterion = nn.MSELoss()
    criterion.cuda()

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=opts.learning_rate, weight_decay=opts.l2
    )
    if opts.no_val:
        scheduler = StepLR(optimizer, step_size=12, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    model.train()
    training_losses = []
    for n_iter in range(opts.num_iters):
        logits = model(
            data["train"]["discretized_spatial_feature"] if opts.spatial else None,
            data["train"]["appearance_feature"] if opts.appr else None,
            data["train"]["predicate"],
        )
        loss = criterion(logits, data["train"]["label"])
        training_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n_iter % opts.val_interval == (opts.val_interval - 1):
            if opts.no_val:
                scheduler.step()
            else:
                print("training loss = %f" % np.mean(training_losses[-20:]))
                model.eval()
                logits = model(
                    data["valid"]["discretized_spatial_feature"]
                    if opts.spatial
                    else None,
                    data["valid"]["appearance_feature"] if opts.appr else None,
                    data["valid"]["predicate"],
                )
                loss = criterion(logits, data["valid"]["label"])
                print("validation loss = %f" % loss)
                correct = (logits > 0) * (data["valid"]["label"] > 0) + (
                    logits <= 0
                ) * (data["valid"]["label"] <= 0)
                for i, predicate in enumerate(predicate_categories):
                    indexes = data["valid"]["predicate"][:, i]
                    acc = correct[indexes].sum().item() / indexes.sum().item()
                    print("%s = %f" % (predicate, acc))
                acc = correct.sum().item() / logits.size(0)
                print("validation accuracy = %f" % acc)

                scheduler.step(acc)
                model.train()

    print("testing..")
    model.eval()
    logits = model(
        data["test"]["discretized_spatial_feature"] if opts.spatial else None,
        data["test"]["appearance_feature"] if opts.appr else None,
        data["test"]["predicate"],
    )
    correct = (logits > 0) * (data["test"]["label"] > 0) + (logits <= 0) * (
        data["test"]["label"] <= 0
    )
    for i, predicate in enumerate(predicate_categories):
        indexes = data["test"]["predicate"][:, i]
        acc = correct[indexes].sum().item() / indexes.sum().item()
        print("%s = %f" % (predicate, acc))
    acc = correct.sum().item() / logits.size(0)
    print("testing accuracy = %f" % acc)
    pickle.dump((data["test"]["_id"], logits), open("pred.pickle", "wb"))
