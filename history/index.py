import os

import pandas as pd

INDEX_PATH = r"D:\petrtsv\projects\ds\torch-nn-project\index.csv"
NAME_COL = 'Name'

index_df = None


def save_df():
    index_df.to_csv(INDEX_PATH)


if os.path.exists(INDEX_PATH):
    index_df = pd.read_csv(INDEX_PATH, index_col=0)
else:
    index_df = pd.DataFrame([{NAME_COL: 'Null', 'Explanation': 'To make data frame not empty'}])
    save_df()


def save_record(name, **kwargs):
    kwargs[NAME_COL] = name
    record_df = pd.DataFrame([kwargs])

    global index_df
    index_df = index_df.append(record_df, sort=False, ignore_index=True)

    save_df()
