import pickle
from sklearn.model_selection import train_test_split

# load preprocessed dataset
with open('./dataset/prep_data', 'rb') as fp:
    data = pickle.load(fp)
# split train and validation data
train, valid = train_test_split(data, test_size=0.2)
# save train dataset and validation dataset
with open('./dataset/train', 'wb') as fp:
    pickle.dump(train, fp, protocol=2)
with open('./dataset/valid', 'wb') as fp:
    pickle.dump(valid, fp, protocol=2)

print 1
""" ## make tokens file 
# load preprocessed dataset

with open('./dataset/dataset2', 'rb') as fp:
    data = pickle.load(fp)

tokens = []
for song in data:
    for token in song:
      
        ## count the number of times tokens appear
        #idx = tokens_ref.index(token)
        #cnt_tokens[idx] += 1
        
        if token not in tokens:
            tokens.append(token)
    print('tokens:', tokens)
    print('len(tokens):', len(tokens))
# save file
with open('./dataset/tokens', 'wb') as fp:
    pickle.dump(tokens, fp, protocol=2)
print('a')
"""

## tokenize
with open('./dataset/dataset2', 'rb') as fp:
    data = pickle.load(fp)

# load list of unique tokens
with open('./dataset/tokens', 'rb') as fp:
    tokens_ref = pickle.load(fp)

batch = []
sequence = []
for song in data:
    for token in song:
        idx = tokens_ref.index(token)
        if len(sequence) < 100:
            sequence.append(idx)
        else:
            batch.append(sequence)
            sequence = []

# save file
with open('./dataset/prep_data', 'wb') as fp:
    pickle.dump(batch, fp, protocol=2)
print('a')