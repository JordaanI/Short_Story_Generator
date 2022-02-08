def gen_seq(data,column,base_length):
    training = []
    labels = []

    base_length = base_length
    seq_length = base_length*2

    data = data.drop(data[data[column].map(len) < seq_length].index)

    #CUtting the tokens into sequences and adding them to an array, here every 36 words in our token sequence forms
    # a training label pair from the begging to the end of the token sequence. 

    lengths = [len(sequence) for sequence in data[column]]
    if min(lengths) >= seq_length:
        for sequence in data[column]:
            for i in range(seq_length, len(sequence)):
                cut = sequence[i - seq_length:i + 1]
                training.append(cut[:-1])
                labels.append(cut[-1])
    else: #Not expected to be used but here to avoid an error
        print(f'The sequence at {lengths.index(min(lengths))} is too short.')
    
    return training, labels

def train_test_split(training, labels, num_words):
    import numpy as np

    compact = list(zip(training,labels))
    np.random.shuffle(compact)
    training, labels = zip(*compact)

    #split into 75% training to 25% test

    X_train = np.array(training[:int(0.75*len(training))])
    X_test = np.array(training[int(0.75*len(training)):])

    y_train_base = np.array(labels)[:int(0.75*len(labels))]
    y_test_base = np.array(labels)[int(0.75*len(labels)):]

    y_train = np.zeros((len(y_train_base), num_words), dtype=np.int8)
    y_test = np.zeros((len(y_test_base), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(y_train_base):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(y_test_base):
        y_test[example_index, word_index] = 1

    print(f'The training sequence shape is {X_train.shape}, the training label shape is {y_train.shape}')
    print(f'The test sequence shape is {X_test.shape}, the test label shape is  {y_test.shape}')

    return X_train, y_train, X_test, y_test