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

def make_model(num_words,
                          embedding_matrix,
                          lstm_cells=64,
                          trainable=False,
                          lstm_layers=1,
                          bi_direc=False):

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional, Dense, Embedding

    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1))

    # Add final LSTM cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_cells,
                    return_sequences=False,
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1))
    model.add(Dense(128, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def lookup(param, word_index, word_lexicon):
    if type(param) is int:
        return word_index[param]
    elif type(param) is str:
        return word_lexicon[param]
    else:
        print('Parameter not accepted.')

def generate_sequence(X_test,word_index,word_lexicon,model):

    import numpy as np
    import random

    training_length = 35
    seq = list(random.choice(X_test))
    diversity=1

    seed_idx = 0
    end_idx = training_length

    seed = list(seq[seed_idx:end_idx])
    original_sequence = [word_index[i] for i in seed]
    generated = seed[:] + ['#']


    actual = generated[:] + seq[end_idx:end_idx + training_length]

    for i in range(training_length):
        preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
            np.float64)
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)[0]
        next_idx = np.argmax(probas)
        seed = seed[1:] + [next_idx]
        generated.append(next_idx)

    SEED = ' '.join(original_sequence)
    AI = ' '.join([lookup(i, word_index, word_lexicon) for i in generated[36:]])
    REAL = ' '.join([lookup(i, word_index, word_lexicon) for i in actual[36:]])
    return 'Seeded sequence: ' + SEED + '\n Actual sequence:' + REAL + '\n Generated squence:' + AI 