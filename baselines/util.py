from gensim.models import KeyedVectors
import json
import pickle
import numpy as np
from autocorrect import spell
import cv2
import os
import random
from collections import defaultdict
import torch


if not "NO_WORD2VEC" in os.environ:
    print(" => loading the word2vec model..")
    word2vec = KeyedVectors.load_word2vec_format(
        "GoogleNews-vectors-negative300.bin.gz", binary=True, unicode_errors="ignore"
    )
else:
    word2vec = defaultdict(lambda: np.zeros((300,), dtype=np.float32))
    print("WARNING: WORD2VEC IS NOT LOADED!")


def phrase2vec(phrase, max_phrase_len, word_embedding_dim):
    vec = np.zeros((max_phrase_len, word_embedding_dim,), dtype=np.float32)
    for i, word in enumerate(phrase.split()):
        assert i < max_phrase_len
        if word in word2vec:
            vec[i] = word2vec[word]
        elif spell(word) in word2vec:
            vec[i] = word2vec[spell(word)]
        else:
            pass
            # print(word)
    return vec


def onehot(k, n):
    encoding = np.zeros((n,), dtype=np.float32)
    encoding[k] = 1.0
    return encoding


def read_img(url, imagepath):
    if url.startswith("http"):  # flickr
        filename = os.path.join(imagepath, "flickr", url.split("/")[-1])
    else:  # nyu
        filename = os.path.join(imagepath, "nyu", url.split("/")[-1])
    img = cv2.imread(filename).astype(np.float32, copy=False)[:, :, ::-1]
    assert img.shape[2] == 3
    return img


def accuracies(pred_file, gt_file, split, vis, args):
    gt = {}
    data = json.load(open(gt_file))
    for img in data:
        if img["split"] != split:
            continue
        for annot in img["annotations"]:
            annot["url"] = img["url"]
            gt[annot["_id"]] = annot

    cnts = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    _ids, predictions = pickle.load(open(pred_file, "rb"))
    for _id, prediction in zip(_ids, predictions):
        predicate = gt[_id]["predicate"]
        if (prediction > 0.0) == gt[_id]["label"]:
            cnts[predicate]["correct"] += 1
            cnts["overall"]["correct"] += 1
        else:
            cnts[predicate]["incorrect"] += 1
            cnts["overall"]["incorrect"] += 1

    accs = {}
    for k, v in cnts.items():
        accs[k] = 100.0 * v["correct"] / (v["correct"] + v["incorrect"])
    return accs


def num_true_positives(logits, labels):
    return torch.sum(torch.eq(torch.gt(logits, 0.0).float(), labels).float()).item()
