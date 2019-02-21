from __future__ import print_function

import argparse
from collections import Counter
from math import ceil
import pickle
import molecule_gdb9_vae
import numpy as np
from rdkit import Chem
from tqdm import tqdm

BATCH = 256


def encode_and_decode(smi, n_encode, n_decode):
    decoded_molecules = {s: [] for s in smi}
    assert len(decoded_molecules) == len(smi), 'duplicate SMILES'
    with tqdm(total=n_encode * n_decode) as pbar:
        for i in range(n_encode):
            latent_vec = grammar_model.encode(smi)
            latent_vec = np.asarray(latent_vec, dtype=np.float32)
            assert latent_vec.shape[0] == len(smi)
            for j in range(n_decode):
                decoded_smi = grammar_model.decode(latent_vec)
                assert len(decoded_smi) == len(smi)
                for s, r in zip(smi, decoded_smi):
                    decoded_molecules[s].append(r)
                pbar.update()
    return decoded_molecules


def norm_smi(smi):
   mol = Chem.MolFromSmiles(smi)
   if mol is not None:
       return Chem.MolToSmiles(mol, isomericSmiles=True)


def num_identical(smi_ref, smi_others):
   c = Counter()
   c.update(smi_others)
   num = c[smi_ref]
   return num


parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True,
                    help='Path to HDF5 file with weights.')
parser.add_argument('--smiles', required=True,
                    help='Path to files with SMILES string to embed and reconstruct.')
parser.add_argument('-o', '--output', required=True,
                    help='Path to output file.')
parser.add_argument('--num_samples', type=int)
parser.add_argument('--num_encode', type=int, default=10)
parser.add_argument('--num_decode', type=int, default=100)

args = parser.parse_args()

# 1. load grammar VAE
grammar_model = molecule_gdb9_vae.Gdb9GrammarModel(args.weights,
    latent_rep_size=64)

# 2. load SMILES strings
smiles = []
with open(args.smiles) as fin:
    for line in fin:
        smi, _ = line.strip().split("\t", 1)
        smiles.append(smi)

idx = np.arange(len(smiles), dtype=int)
if args.num_samples is not None:
    idx = np.random.choice(idx, size=args.num_samples, replace=False)

all_decoded = None
num_batches = int(ceil(len(idx) / float(BATCH)))
print('Encoding/Decoding %d molecules in %d batches' % (len(idx), num_batches))

# 3. encode and decode some SMILES strings
for k in range(num_batches):
    start = k * BATCH
    end = min(start + BATCH, len(idx))
    smiles_batch = [smiles[i] for i in idx[start:end]]

    decoded = encode_and_decode(smiles_batch, args.num_encode, args.num_decode)
    if all_decoded is None:
        all_decoded = decoded
    else:
        all_decoded.update(decoded)

with open(args.output, 'wb') as fout:
    pickle.dump(all_decoded, fout)

accuracy = []
for k, v in all_decoded.items():
    acc = num_identical(k, v) / len(v)
    accuracy.append(acc)

print(np.mean(accuracy), np.std(accuracy, ddof=1))