def id2png(_id):
    return _id + '.png'


def get_pred(x, threshold=0.5):
    return x[0] > threshold
