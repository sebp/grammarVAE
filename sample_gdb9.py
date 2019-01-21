from contextlib import contextmanager
import molecule_gdb9_vae
import numpy as np
from rdkit import Chem, rdBase
from tqdm import trange


@contextmanager
def disable_rdkit_log():
    rdBase.DisableLog('rdApp.error')
    try:
        yield
    finally:
        rdBase.EnableLog('rdApp.error')


def decode_from_latent_space(z, model):
    success = False
    smiles = set()
    for _ in range(10):
        try:
            smiles.update(model.decode(z))
        except:
            pass

    return check_smiles(smiles)


def check_smiles(smiles):
    with disable_rdkit_log():
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue

            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            yield smi


# 1. load grammar VAE
grammar_weights = "results/gdb9_vae_grammar_L56_E100_val.hdf5"
grammar_model = molecule_gdb9_vae.Gdb9GrammarModel(grammar_weights)

n_samples = 10000
batch_size = 100
latent_rep_size = 56
rnd = np.random.RandomState(8793)

# mol: decoded SMILES string
# NOTE: decoding is stochastic so calling this function many
# times for the same latent point will return different answers

epochs = n_samples // batch_size

fout = open("generated.smi", "w")
for _ in trange(epochs):
    z1 = rnd.randn(batch_size, latent_rep_size).astype(np.float32)
    for smi in decode_from_latent_space(z1, grammar_model):
        if smi is not None:
            fout.write(smi)
            fout.write("\n")
fout.close()
