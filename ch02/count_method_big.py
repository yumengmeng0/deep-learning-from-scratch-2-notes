import sys
sys.path.append('..')

import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


# [query] you
#  i: 0.6873486042022705
#  we: 0.6131157875061035
#  somebody: 0.5398598909378052
#  've: 0.537065327167511
#  anybody: 0.5356476306915283
# [query] year
#  quarter: 0.6474596261978149
#  month: 0.6340356469154358
#  last: 0.6261981129646301
#  next: 0.6050572395324707
#  third: 0.5678852200508118
# [query] car
#  luxury: 0.6676775217056274
#  auto: 0.6284433603286743
#  lexus: 0.5267549753189087
#  domestic: 0.5098328590393066
#  vehicle: 0.4968051016330719
# [query] toyota
#  motor: 0.6990517377853394
#  motors: 0.6611846089363098
#  mazda: 0.6426794528961182
#  nissan: 0.6354326009750366
#  honda: 0.6294161081314087
