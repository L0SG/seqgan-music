import cPickle
from sklearn.model_selection import train_test_split
# load preprocessed dataset
with open('./dataset/dataset_20', 'rb') as fp:
    data = cPickle.load(fp)
# split train and validation data
train, valid = train_test_split(data, test_size=0.2)
# save train dataset and validation dataset
with open('train', 'wb') as fp:
    cPickle.dump(train, fp, protocol=2)
with open('valid', 'wb') as fp:
    cPickle.dump(valid, fp, protocol=2)


# load list of unique tokens
# with open('./dataset/tokens', 'rb') as fp:
#     tokens_ref = cPickle.load(fp)
#tokens = []
#cnt_tokens = [0] * len(tokens_ref)
batch = []
sequence = []
for song in data:
    for token in song:
        """ # count the number of times tokens appear
        idx = tokens_ref.index(token)
        cnt_tokens[idx] += 1
        #if token not in tokens:
        #    tokens.append(token)
        """
        idx = tokens_ref.index(token)
        if len(sequence) < 20:
            sequence.append(idx)
        else:
            batch.append(sequence)
            sequence = []

print('a')
a=3
""" # save file
with open('dataset2', 'wb') as fp:
    cPickle.dump(data2, fp, protocol=2)
"""