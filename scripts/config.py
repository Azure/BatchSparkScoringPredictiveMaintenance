import argparse


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, "w") as f:
        print(
            'Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals())
        )
        s = s.replace(old_string, new_string)
        f.write(s)


# Module sys has to be imported:
import sys

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))

#

parser = argparse.ArgumentParser(
    description="searches the jobs folder for Databricks Jobs json files,\n"
    + "and configures the scripts to connect to the clusterID and username provided.\n"
    + ""
)
parser.add_argument(
    "clusterID",
    metavar="--clusterID",
    type=str,
    nargs="+",
    help="The cluster ID is found using the databricks CLI command:\n"
    + "\ndatabricks cluster list\n",
)
parser.add_argument(
    "username",
    metavar="--username",
    type=str,
    nargs="+",
    help="Found from the Databricks Workspace and should look like\n"
    + "your active directory email address.",
)

args = parser.parse_args()

print(args.clusterID)
print(args.username)
