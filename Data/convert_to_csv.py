import hdf5plugin 
import h5py
import pandas as pd

with h5py.File('train_cite_inputs.h5', 'r') as f:
    cell_barcodes = f['train_cite_inputs/axis1'][:].astype(str)
    X             = f['train_cite_inputs/block0_values'][:]

meta = pd.read_csv('metadata.csv', index_col=0)
cite_meta = meta[meta['technology'] == 'citeseq']

day_map = {2: 1, 3: 2, 4: 3, 7: 4}

n_genes = X.shape[1]
df = pd.DataFrame(X, index=cell_barcodes, columns=[f'd{i+1}' for i in range(n_genes)])
df.insert(0, 'sample', cite_meta.loc[cell_barcodes, 'day'].map(day_map).values)
df = df.dropna(subset=['sample'])
df['sample'] = df['sample'].astype(int)

df.to_csv('citeseq_22.csv', index=False)
print(df.shape)