"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""

import pandas as pd
from pandas import DataFrame as df

data = pd.read_csv('data/data.tsv', sep='\t')

train = data.sample(frac=0.64, random_state=74)
print(train['label'].value_counts())
valid_test = data.drop(train.index)

valid = valid_test.sample(frac=16/36, random_state=30)
print(valid['label'].value_counts())
test = valid_test.drop(valid.index)
print(test['label'].value_counts())

overfit = train.sample(frac=50/6400, random_state=19)
print(overfit['label'].value_counts())

train.to_csv('data/train.tsv', sep='\t', index=False)
valid.to_csv('data/validation.tsv', sep='\t', index=False)
test.to_csv('data/test.tsv', sep='\t', index=False)
overfit.to_csv('data/overfit.tsv', sep='\t', index=False)