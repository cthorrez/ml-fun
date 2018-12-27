import torch
from noise import OrnsteinUhlenbeck
import matplotlib.pyplot as plt

def main():
    ou = OrnsteinUhlenbeck(mu=torch.zeros(1), sigma=0.05 * torch.ones(1))

    xs = list(range(100000))
    ys = []
    for x in xs:
        y = ou()
        ys.append(y.data)

    plt.plot(xs,ys)
    plt.show()




if __name__ == '__main__':
    main()