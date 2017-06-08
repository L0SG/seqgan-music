import cPickle
from sklearn.model_selection import train_test_split

with open('dataset/data_reference.txt', 'r') as fp:
    token_stream = []
    for line in fp:
        line = line.strip()
        line = line.split()
        parse_line = [int(x) for x in line]
        if len(parse_line) == 20:
            token_stream.append(parse_line)
train_ref, valid_ref = train_test_split(token_stream, test_size=0.2)

with open('dataset/train_ref', 'wb') as fp:
    cPickle.dump(train_ref, fp, protocol=2)
with open('dataset/valid_ref', 'wb') as fp:
    cPickle.dump(valid_ref, fp, protocol=2)
print ('test')