# Visualize the relations in SpatialSense
import json
from PIL import Image, ImageDraw, ImageFont
import argparse
import random
import os
import pdb

parser = argparse.ArgumentParser(description="Visualize the relations in SpatialSense")
parser.add_argument(
    "--num", type=int, default=10, help="The number of relations to visualize"
)
parser.add_argument(
    "--dataroot",
    default=".",
    help="The directory where annotation.js and images reside",
)
parser.add_argument(
    "--output",
    default="./visualizations",
    help="The directory to output visualizations",
)
parser.add_argument("--seed", default=1, help="The random seed")
args = parser.parse_args()

if os.path.exists(args.output):
    os.system("rm -r " + args.output)
os.makedirs(args.output)

random.seed(args.seed)

relations = []
data = json.load(open("annotations.json"))
for img in data:
    for annot in img["annotations"]:
        annot["url"] = img["url"]
        annot["width"] = img["width"]
        annot["height"] = img["height"]
        relations.append(annot)

relations = random.sample(relations, args.num)


def url2path(url):
    if url.startswith("http"):  # flickr
        return os.path.join(args.dataroot, "images", "flickr", url.split("/")[-1])
    else:  # nyu
        return os.path.join(args.dataroot, "images", "nyu", url.split("/")[-1])


for i, rel in enumerate(relations):
    img = Image.open(url2path(rel["url"]))
    vis = Image.new("RGB", (max(rel["width"], 300), rel["height"] + 40), color="white")
    vis.paste(img)

    draw = ImageDraw.Draw(vis)
    fnt = ImageFont.truetype("arial.ttf", size=25)
    s = "%s-%s-%s    %s" % (
        rel["subject"]["name"],
        rel["predicate"],
        rel["object"]["name"],
        str(rel["label"]),
    )
    draw.text((10, rel["height"]), s, font=fnt, fill="black")

    y0, y1, x0, x1 = rel["subject"]["bbox"]
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    y0, y1, x0, x1 = rel["object"]["bbox"]
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

    vis.save(os.path.join(args.output, "%05d.jpg" % i))

print("Visualizations saved to", args.output)
