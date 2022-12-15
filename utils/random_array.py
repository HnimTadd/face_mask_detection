from lib import np

def union_shuffled_copies(a, b):
    assert len(a)== len(b)
    p = np.random.permutation(len(a))
    return np.array(a)[p], np.array(b)[p]

def train_val_separate(imgs, anns, ratio):
    assert len(imgs) == len(anns)
    return imgs[:int(len(imgs)*ratio)], anns[:int(len(anns)*ratio)], imgs[int(len(imgs) *ratio) :], anns[int(len(anns)*ratio) :] 