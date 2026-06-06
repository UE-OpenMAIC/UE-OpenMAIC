import numpy as np

def reorder_label(labels):

    labels = np.asarray(labels)

    mapping = {}

    out = []

    nxt = 0

    for x in labels.tolist():

        if x not in mapping:

            mapping[x] = nxt

            nxt += 1

        out.append(mapping[x])

    return np.array(out, dtype=int)
