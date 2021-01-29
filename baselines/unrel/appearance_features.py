import pdb
import random
import torch
import torch.nn as nn
import os
import numpy as np
from progressbar import ProgressBar
import cv2
import json
import pickle
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA


def read_img(url):
    if url.startswith("http"):  # flickr
        filename = os.path.join("../images/flickr", url.split("/")[-1])
    else:  # nyu
        filename = os.path.join("../images/nyu", url.split("/")[-1])
    img = cv2.imread(filename).astype(np.float32, copy=False)[:, :, ::-1]
    assert img.shape[2] == 3
    return img


vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = nn.Sequential(*vgg16.classifier)[:-3]
vgg16.cuda()
vgg16.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess(img_crop):
    img_crop = torch.Tensor(
        cv2.resize(img_crop, (224, 224)).transpose([2, 0, 1])
    ).cuda()
    img_crop /= 255.0
    img_crop = normalize(img_crop)
    return img_crop


def raw_appearance_feature(url, bbox_s, bbox_o):
    img = read_img(url)
    img_s = preprocess(img[bbox_s[0] : bbox_s[1], bbox_s[2] : bbox_s[3]])
    img_o = preprocess(img[bbox_o[0] : bbox_o[1], bbox_o[2] : bbox_o[3]])
    fea = vgg16(torch.stack([img_s, img_o]))
    fea_s = fea[0] / fea[0].norm()
    fea_o = fea[1] / fea[1].norm()
    return fea_s.detach().cpu().numpy(), fea_o.detach().cpu().numpy()


if __name__ == "__main__":
    data = json.load(open("../annotations.json"))
    X = []
    bar = ProgressBar(max_value=len(data))
    for i, img in enumerate(data):
        if img["split"] == "test":
            continue
        for annot in img["annotations"]:
            X.extend(
                raw_appearance_feature(
                    img["url"], annot["subject"]["bbox"], annot["object"]["bbox"]
                )
            )
        bar.update(i)

    random.shuffle(X)
    X = np.vstack(X)
    pca = PCA(n_components=300)
    pca.fit(X)
    pickle.dump(pca, open("pca.pickle", "wb"))
