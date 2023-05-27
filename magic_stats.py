import numpy as np
import pandas as pd
import argparse
from os import walk, sep


def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('source_dir',
                        type=str,
                        help='A required integer positional argument')
    args = parser.parse_args()

    dfs = {}
    for (dirpath, dirname, filenames) in walk(args.source_dir):
        for filename in filenames:
            if filename == "metrics.csv":
                name = dirpath.split(sep)[-1]
                dfs[name] = pd.read_csv(sep.join([dirpath, filename]))

    for key, df in dfs.items():
        print(f'---{key}---')
        print(df.describe())
        print()


if __name__ == "__main__":
    main()
