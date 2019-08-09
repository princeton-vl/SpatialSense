import pdb
import json
import pickle
import numpy as np
import math
import random
from sklearn.mixture import GaussianMixture


def raw_spatial_feature(bbox_s, bbox_o):
    w_s = bbox_s[3] - bbox_s[2]
    h_s = bbox_s[1] - bbox_s[0]
    w_o = bbox_o[3] - bbox_o[2]
    h_o = bbox_o[1] - bbox_o[0]

    # Scale
    scale_s = w_s * h_s;
    scale_o = w_o * h_o;

    # Offset
    xc_s = (bbox_s[2] + bbox_s[3]) / 2.
    yc_s = (bbox_s[0] + bbox_s[1]) / 2.
    xc_o = (bbox_o[2] + bbox_o[3]) / 2.
    yc_o = (bbox_o[0] + bbox_o[1]) / 2.
    offsetx = xc_o - xc_s
    offsety = yc_o - yc_s

    # Aspect ratio
    aspect_s = w_s / h_s;
    aspect_o = w_o / h_o;

    # Overlap
    boxI_xmin = max(bbox_s[2], bbox_o[2])
    boxI_ymin = max(bbox_s[0], bbox_o[0])
    boxI_xmax = min(bbox_s[3], bbox_o[3])
    boxI_ymax = min(bbox_s[1], bbox_o[1])
    wI = max(boxI_xmax - boxI_xmin, 0)
    yI = max(boxI_ymax - boxI_ymin, 0)
    areaI = wI * yI
    areaU = scale_s + scale_o - areaI

    # Fill the raw spatial feature
    feature = np.asarray([offsetx / math.sqrt(scale_s), 
                          offsety / math.sqrt(scale_s), 
                          math.sqrt(scale_o / scale_s), 
                          aspect_s, 
                          aspect_o, 
                          math.sqrt(areaI / areaU)])
    return feature


if __name__ == '__main__':
    data = json.load(open('../annotations.json'))
    X = []
    for img in data:
        if img['split'] == 'test':
            continue
        for annot in img['annotations']:
            X.append(raw_spatial_feature(annot['subject']['bbox'], annot['object']['bbox']))     
    
    random.shuffle(X)
    X = np.vstack(X)
    gmm = GaussianMixture(400, max_iter=100, verbose=1)
    gmm.fit(X)
    pickle.dump(gmm, open('gmm.pickle', 'wb'))
