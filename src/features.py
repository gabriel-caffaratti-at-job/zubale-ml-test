# src/features.py
import os
import pickle
import pandas as pd

class TabPreprocess:
    """
    Pre-processor for tabular data
    - Transform categorigal variables to indices for embedings.
    - complete numerical empty values with mean of the column.
    - Save data (means and mappings) to artifacts_dir/preprocessor.pkl
    - Load data from artifacts_dir/preprocessor.pkl
    """
    def __init__(self, cat_cols, num_cols, artifacts_dir="artifacts"):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.artifacts_dir = artifacts_dir
        self.means = {}
        self.cat_maps = {}

    def fit(self, df: pd.DataFrame):
        # calculate means for numerical columns
        self.means = df[self.num_cols].astype("float32").mean(numeric_only=True).to_dict()
        # create mapping for categorical columns
        self.cat_maps = {}
        for c in self.cat_cols:
            uniques = pd.Series(df[c].dropna().unique()).tolist()
            self.cat_maps[c] = {v: i+1 for i, v in enumerate(uniques)}
        # save artifacts
        os.makedirs(self.artifacts_dir, exist_ok=True)
        with open(os.path.join(self.artifacts_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump({
                "means": self.means,
                "cat_maps": self.cat_maps,
                "cat_cols": self.cat_cols,
                "num_cols": self.num_cols
            }, f)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # categorical to indices
        for c in self.cat_cols:
            out[c] = out[c].map(self.cat_maps[c]).fillna(0).astype("int64")
        # fill empty numerical with mean
        for c in self.num_cols:
            out[c] = out[c].astype("float32").fillna(self.means[c])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    @classmethod
    def load(cls, artifacts_dir="artifacts"):
        with open(os.path.join(artifacts_dir, "preprocessor.pkl"), "rb") as f:
            blob = pickle.load(f)
        obj = cls(blob["cat_cols"], blob["num_cols"], artifacts_dir)
        obj.means = blob["means"]
        obj.cat_maps = blob["cat_maps"]
        return obj
