from datasets import load_dataset
import pandas as pd
from pathlib import Path
import numpy as np


DEFAULT_COLUMNS = ["participant","task","trial"]

def load_dataframe(path,columns:list[str]|None=None):
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        path = str(path)
        suffix = Path(path).suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(path)
        elif suffix == '.parquet' or path.startswith("hf://"):
            df = pd.read_parquet(path)
        elif suffix == '.jsonl':
            df = pd.read_json(path, lines=True)
        elif suffix == '.json':
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    if columns is not None:
        columns = list(dict.fromkeys(DEFAULT_COLUMNS+columns))
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        return df.loc[:, columns]

    return df


def load_hf_dataset(source:str, split:str, columns:list[str], **kwargs):
    ds = load_dataset(source, split=split, **kwargs)
    if columns is not None:
        missing = [col for col in columns if col not in ds.column_names]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        
        ds = ds.select_columns(columns)
        
    return ds
    


