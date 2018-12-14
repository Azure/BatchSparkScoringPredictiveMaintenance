import argparse
import os
import glob
import sys

parser = argparse.ArgumentParser(
    description="searches the jobs folder for Databricks Jobs json files,\n"
    + "and configures the scripts to connect to the clusterID and username provided."
)
parser.add_argument(
    "-c",
    "--clusterID",
    type=str,
    help="The cluster ID is found using the databricks CLI command:\n"
    + "\ndatabricks clusters list\n",
)
parser.add_argument(
    "-u",
    "--username",
    type=str,
    help="Found from the Databricks Workspace and should look like\n"
    + "your active directory email address.",
)
parser.add_argument("infile", type=str)

args = parser.parse_args()

print(args.clusterID)
print(args.username)
print(args.infile)


for filename in glob.glob(os.path.join(args.infile, "*.json")):
    with open(filename, "r") as f:
        s = f.read()
        if "<clusterid>" in s:
            s = s.replace("<clusterid>", args.clusterID)
        if "<uname@example.com>" in s:
            s = s.replace("<uname@example.com>", args.username)

    fname = filename.replace("tmpl", "json")
    with open(fname, "w") as fs:
        fs.write(s)

