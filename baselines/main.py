import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import progressbar
from options import parse_args
from dataloader import create_dataloader
from models.recurrent_phrase_encoder import RecurrentPhraseEncoder
import pickle
import os
import sys
from models.vtranse import VtransE
from models.drnet import DRNet
from models.vipcnn import VipCNN
from models.pprfcn import PPRFCN
import shutil
from util import accuracies, num_true_positives


def train(model, criterion, optimizer, loader, epoch, args):
    model.train()
    loss = 0.0
    acc = 0.0
    num_samples = 0

    bar = progressbar.ProgressBar(max_value=len(loader))
    for idx, data_batch in enumerate(loader):
        subj_batch_var = data_batch["subject"]["embedding"].cuda()
        obj_batch_var = data_batch["object"]["embedding"].cuda()
        predicate = data_batch["predicate"].cuda()
        label_batch_var = torch.squeeze(data_batch["label"]).cuda()

        if args.model == "drnet":
            img = data_batch["bbox_img"].cuda()
            mask_batch_var = data_batch["bbox_mask"].cuda()
            output = model(
                subj_batch_var, obj_batch_var, img, mask_batch_var, predicate
            )
        elif args.model == "vtranse":
            img = data_batch["full_img"].cuda()
            ts_batch_var = data_batch["subject"]["t"].cuda()
            to_batch_var = data_batch["object"]["t"].cuda()
            bboxs_batch_var = data_batch["subject"]["bbox"].cuda()
            bboxo_batch_var = data_batch["object"]["bbox"].cuda()
            output = model(
                subj_batch_var,
                obj_batch_var,
                img,
                ts_batch_var,
                to_batch_var,
                bboxs_batch_var,
                bboxo_batch_var,
                predicate,
            )
        elif args.model == "vipcnn" or args.model == "pprfcn":
            img = data_batch["full_img"].cuda()
            bbox_s = data_batch["subject"]["bbox"].cuda()
            bbox_o = data_batch["object"]["bbox"].cuda()
            output = model(img, bbox_s, bbox_o, predicate)

        loss_batch_var = criterion(output, label_batch_var)

        loss_batch = loss_batch_var.item()
        loss += len(data_batch["label"]) * loss_batch
        acc += num_true_positives(output, label_batch_var)
        num_samples += len(data_batch["label"])

        optimizer.zero_grad()
        loss_batch_var.backward()
        optimizer.step()

        bar.update(idx)

    loss /= num_samples
    acc /= num_samples / 100.0

    return loss, acc


def test(split, model, criterion, loader, epoch, args):
    model.eval()
    loss = 0.0
    _ids = []
    predictions = []

    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(loader))
        for idx, data_batch in enumerate(loader):
            _ids.extend(data_batch["_id"])
            subj_batch_var = data_batch["subject"]["embedding"].cuda()
            obj_batch_var = data_batch["object"]["embedding"].cuda()
            predicate = data_batch["predicate"].cuda()
            if args.model == "drnet":
                img = data_batch["bbox_img"].cuda()
                mask_batch_var = data_batch["bbox_mask"].cuda()
                output = model(
                    subj_batch_var, obj_batch_var, img, mask_batch_var, predicate
                )
            elif args.model == "vtranse":
                img = data_batch["full_img"].cuda()
                ts_batch_var = data_batch["subject"]["t"].cuda()
                to_batch_var = data_batch["object"]["t"].cuda()
                bboxs_batch_var = data_batch["subject"]["bbox"].cuda()
                bboxo_batch_var = data_batch["object"]["bbox"].cuda()
                output = model(
                    subj_batch_var,
                    obj_batch_var,
                    img,
                    ts_batch_var,
                    to_batch_var,
                    bboxs_batch_var,
                    bboxo_batch_var,
                    predicate,
                )
            elif args.model == "vipcnn" or args.model == "pprfcn":
                img = data_batch["full_img"].cuda()
                bbox_s = data_batch["subject"]["bbox"].cuda()
                bbox_o = data_batch["object"]["bbox"].cuda()
                output = model(img, bbox_s, bbox_o, predicate)

            predictions.append(output)

            if "label" in data_batch:
                label_batch_var = torch.squeeze(data_batch["label"]).cuda()
                loss_batch_var = criterion(output, label_batch_var)
                loss_batch = loss_batch_var.item()
                loss += len(data_batch["label"]) * loss_batch

            bar.update(idx)

        if epoch is None:
            epoch = "test"
        predictions = [v.item() for v in torch.cat(predictions)]
        if split == "valid":
            pred_file = os.path.join(
                args.log_dir, "predictions/pred_%02d.pickle" % epoch
            )
        else:
            pred_file = os.path.join(args.log_dir, "predictions/pred_test.pickle")
        pickle.dump((_ids, predictions), open(pred_file, "wb"))
        accs = accuracies(pred_file, args.datapath, split, split == "test", args)
        return loss, accs


def main():
    args = parse_args()

    dataloader_train = create_dataloader(args.train_split, True, args)
    dataloader_valid = create_dataloader("valid", True, args)
    dataloader_test = create_dataloader("test", True, args)
    print("%d batches of training examples" % len(dataloader_train))
    print("%d batches of validation examples" % len(dataloader_valid))
    print("%d batches of testing examples" % len(dataloader_test))

    phrase_encoder = RecurrentPhraseEncoder(300, 300)
    if args.model == "drnet":
        model = DRNet(phrase_encoder, args.feature_dim)
    elif args.model == "vtranse":
        model = VtransE(
            phrase_encoder, args.visual_feature_size, args.predicate_embedding_dim
        )
    elif args.model == "vipcnn":
        model = VipCNN(roi_size=args.roi_size, backbone=args.backbone)
    else:
        model = PPRFCN(backbone=args.backbone)
    model.cuda()
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    criterion.cuda()

    optimizer = torch.optim.RMSprop(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.l2,
    )
    if args.train_split == "train":
        scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=args.patience, gamma=0.1)

    start_epoch = 0
    if args.resume != None:
        print(" => loading model checkpoint from %s.." % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        model.cuda()
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

    best_acc = -1.0

    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        print("epoch #%d" % epoch)

        print("training..")
        loss, acc = train(model, criterion, optimizer, dataloader_train, epoch, args)
        print("\n\ttraining loss = %.4f" % loss)
        print("\ttraining accuracy = %.3f" % acc)

        if args.train_split != "train_valid":
            print("validating..")
            loss, accs = test("valid", model, criterion, dataloader_valid, epoch, args)
            print("\n\tvalidation loss = %.4f" % loss)
            print("\tvalidation accuracy = %.3f" % accs["overall"])
            for predi in accs:
                if predi != "overall":
                    print("\t\t%s: %.3f" % (predi, accs[predi]))

        checkpoint_filename = os.path.join(
            args.log_dir, "checkpoints/model_%02d.pth" % epoch
        )
        model.cpu()
        torch.save(
            {
                "epoch": epoch + 1,
                "args": args,
                "state_dict": model.state_dict(),
                "accuracy": acc,
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_filename,
        )
        model.cuda()

        if args.train_split != "train_valid" and best_acc < acc:
            best_acc = acc
            shutil.copyfile(
                checkpoint_filename,
                os.path.join(args.log_dir, "checkpoints/model_best.pth"),
            )
            shutil.copyfile(
                os.path.join(args.log_dir, "predictions/pred_%02d.pickle" % epoch),
                os.path.join(args.log_dir, "predictions/pred_best.pickle"),
            )

        if args.train_split == "train":
            scheduler.step(loss)
        else:
            scheduler.step()

    print("testing..")
    loss, accs = test("test", model, criterion, dataloader_test, None, args)
    print("\n\ttesting loss = %.4f" % loss)
    print("\ttesting accuracy = %.3f" % accs["overall"])
    for predi in accs:
        if predi != "overall":
            print("\t\t%s: %.3f" % (predi, accs[predi]))


if __name__ == "__main__":
    main()
