
def load_file(name, train_frac = 0.6, valid_frac = 0.2):
    return categorize_and_split(open_file(name), train_frac, valid_frac)

def open_file(name):
    data = []
    with open(name, 'r') as file:
        while True:
            line = file.readline().strip()
            if not line:
                break
            parts = line.split(',')
            X = parts[:30], Y = parts[30]
            data.append((X, Y))
    shuffle(data)
    return data


def sign(x):
    if x > 0:
        return 1
    return -1

def categorize_and_split(data, train = 0.6, valid = 0.2):
    negative = [x[0] for x in data if x[1] == 0]
    positive = [x[0] for x in data if x[1] == 1]
    shuffle(positive)
    shuffle(negative)

    neg_train = int(len(negative) * train)
    pos_train = int(len(positive) * train)
    neg_valid = int(len(negative) * (train + valid))
    pos_valid = int(len(positive) * (train + valid))
    train_set = [(x, -1) for x in negative[:neg_train]] + [(x, 1) for x in positive[:pos_train]]
    valid_set = [(x, -1) for x in negative[neg_train:neg_valid]] + [(x, 1) for x in positive[pos_train:pos_valid]]
    test_set = [(x, -1) for x in negative[neg_valid:]] + [(x, 1) for x in positive[pos_valid:]]

    shuffle(train_set)
    shuffle(valid_set)
    shuffle(test_set)
    train_set = np.array(train_set)
    valid_set = np.array(valid_set)
    test_set = np.array(test_set)
    return train_set, valid_set, test_set