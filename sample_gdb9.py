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
    for _ in range(5):
        try:
            smiles.update(model.decode(z))
        except:
            pass

    assert len(smiles) > 0
    return check_smiles(smiles)


def check_smiles(smiles):
    with disable_rdkit_log():
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except ValueError:
                    mol = None

            if mol is not None:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            yield smi


# 1. load grammar VAE
grammar_weights = "results/gdb9_vae_grammar_L56_E100_val.hdf5"
grammar_model = molecule_gdb9_vae.Gdb9GrammarModel(grammar_weights)

n_samples = 250000
batch_size = 100
latent_rep_size = 56
rnd = np.random.RandomState(8793)

# mol: decoded SMILES string
# NOTE: decoding is stochastic so calling this function many
# times for the same latent point will return different answers

epochs = n_samples // batch_size

fout = open("gdb9-generated.smi", "w")
i = 0
for _ in trange(epochs):
    z1 = rnd.randn(batch_size, latent_rep_size).astype(np.float32)
    for smi in decode_from_latent_space(z1, grammar_model):
        if smi is None:
            smi = ''
        fout.write("{},{}\n".format(i, smi))
        i += 1
fout.close()
