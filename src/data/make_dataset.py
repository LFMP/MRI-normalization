import os

import pandas as pd


def get_valid_cases(path,label):
  """
  Get the valid cases (image with mask) from the given path.
  """
  cases = {'patient': [],'image': [], 'mask': [], 'class': []}
  for case in os.listdir(path):
    if case.endswith('.bmp'):
      cases['patient'].append(case.split('_')[0])
      cases['class'].append(label)
      cases['image'].append(case)
      mask = case.replace('.bmp', '_mask.png')
      if mask in os.listdir(path):
        cases['mask'].append(mask)
      else:
        cases['mask'].append(None)
  return pd.DataFrame(cases)

def split_dataset(df, split_ratio):
  """
  Split the dataset into training and validation sets.
  """
  train_df = df.groupby('patient').sample(frac=split_ratio, random_state=42)
  val_df = df.drop(train_df.index)
  return train_df, val_df

def get_dataset(path, split_ratio):
  """
  Get the dataset from the given path.
  """
  datasets = []
  for dataset in os.listdir(path):
    df = get_valid_cases(os.path.join(path, dataset), dataset)
    train_df, val_df = split_dataset(df, split_ratio)
    datasets.append((train_df, val_df))
  train_df = pd.concat([train_df for train_df, _ in datasets])
  val_df = pd.concat([val_df for _, val_df in datasets])
  return train_df, val_df