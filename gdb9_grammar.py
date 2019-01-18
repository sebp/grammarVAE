import nltk
import numpy as np
import six
import pdb

# the zinc grammar
with open('zinc_rules.txt') as fin:
    gram = ''.join(fin.readlines())

# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

# this map tells us the rhs symbol indices for each production rule
rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list),D))
count = 0

# this tells us for each lhs symbol which productions rules should be masked
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

# this tells us the indices where the masks are equal to 1
index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:,i]==1)[0][0])
ind_of_ind = np.array(index_array)

max_rhs = max([len(l) for l in rhs_map])

# some rules aren't used in the GDB9 data so we 
# 0 their masks so they can never be selected
not_used = np.array([
    8,  # aliphatic_organic -> 'S'
    9,  # aliphatic_organic -> 'P'
    11, # aliphatic_organic -> 'I'
    12, # aliphatic_organic -> 'Cl'
    13,  # aliphatic_organic -> 'Br'
    17,  # aromatic_organic -> 's'
    29,  # BACH -> charge class
    30,  # BACH -> charge
    31,  # BACH -> class
    42,  # DIGIT -> '6'
    43,  # DIGIT -> '7'
    44,  # DIGIT -> '8'
    49,  # charge -> '-'
    50,  # charge -> '-' DIGIT
    51,  # charge -> '-' DIGIT DIGIT
    52,  # charge -> '+'
    53,  # charge -> '+' DIGIT
    54,  # charge -> '+' DIGIT DIGIT
    58,  # bond -> '/'
    59,  # bond -> '\\'
])
masks[:, not_used] = 0
