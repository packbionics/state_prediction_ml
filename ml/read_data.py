import os
import pandas as pd


def concat():
    out = pd.DataFrame()
    for f in os.listdir('../data/classified'):
        x = pd.read_excel(f'../data/classified/{f}')
        out = pd.concat([out, x], axis=0)
        
    return out.reset_index()
