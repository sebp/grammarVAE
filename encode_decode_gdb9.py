import sys
#sys.path.insert(0, '..')
import molecule_gdb9_vae
import numpy as np

# 1. load grammar VAE
grammar_weights = "results/gdb9_vae_grammar_L56_E100_val.hdf5"
grammar_model = molecule_gdb9_vae.Gdb9GrammarModel(grammar_weights)

# 2. let's encode and decode some example SMILES strings
smiles = [
        "Nc1noc(=O)c(=O)[nH]1",
        "ON=C1C=CC2CC1O2",
        "CCCC(O)(C)C",
        "C1C2CC3N1C23",
        "CC12CC(N1)(C2N)C#N",
        "N#Cc1[nH]c(c(c1)C)C",
        "CC(=O)C(=O)CC1CN1",
        "Fc1ncoc(=O)c1C",
        "N=C1NC(C1O)C1CN1",
        "N=c1cnn(c(n1)F)C",
]

# z: encoded latent points
# NOTE: this operation returns the mean of the encoding distribution
# if you would like it to sample from that distribution instead
# replace line 83 in molecule_vae.py with: return self.vae.encoder.predict(one_hot)
z1 = grammar_model.encode(smiles)

# mol: decoded SMILES string
# NOTE: decoding is stochastic so calling this function many
# times for the same latent point will return different answers

for mol,real in zip(grammar_model.decode(z1),smiles):
    print(mol, real)

