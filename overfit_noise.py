import pickle
from random import randint
from copy import copy

output = []
for i in xrange(100):
    noisy = []
    for j in xrange(100):
        noisy.append(randint(0, 3215))
    output.append(noisy)
with open('./dataset/rand_seq', 'wb') as fp:
    pickle.dump(output, fp, protocol=2)


with open('./dataset/prep_overfit', 'rb') as fp:
    data = pickle.load(fp)

output = []
for i in xrange(100):
    noisy = copy(data[0])
    # remain first token for bleu score
    noise_idx = randint(1, len(noisy)-1)
    noisy[noise_idx] = randint(0, 3215)
    output.append(noisy)

with open('./dataset/train_overfit', 'wb') as fp:
    pickle.dump(output, fp, protocol=2)