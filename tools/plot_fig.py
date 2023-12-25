import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    loss_path = "/home/chenhuanxin/programs/python/graduation-thesis-code/contrastive-learning/logs/2023_12_20_08_48_03ce-contrastive/train_loss.npy"
    loss_data = np.load(loss_path)

    loss_data_gap = loss_data[::1000]
    plt.plot(range(len(loss_data_gap)), loss_data_gap)
    plt.savefig("figs/train_loss.jpg")
    # plt.show()

    plt.figure()
    acc_path = "/home/chenhuanxin/programs/python/graduation-thesis-code/contrastive-learning/logs/2023_12_20_08_48_03ce-contrastive/test_acc.npy"
    acc_data = np.load(acc_path)

    plt.plot(range(len(acc_data)), acc_data)
    plt.savefig("figs/test_acc.jpg")
    # plt.show()