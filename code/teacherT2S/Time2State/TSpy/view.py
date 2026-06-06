import numpy as np

import matplotlib.pyplot as plt

def plot_mts(data, state_seq):

    data = np.asarray(data)

    state_seq = np.asarray(state_seq)

    n, d = data.shape

    fig, axes = plt.subplots(d + 1, 1, figsize=(12, 2 * (d + 1)), sharex=True)

    for i in range(d):

        axes[i].plot(data[:, i], linewidth=0.8)

        axes[i].set_ylabel(f"ch{i+1}")

    axes[-1].plot(state_seq, linewidth=1.0)

    axes[-1].set_ylabel("state")

    axes[-1].set_xlabel("time")

    plt.tight_layout()
