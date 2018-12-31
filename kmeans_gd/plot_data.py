import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.genfromtxt('data/s1.txt')
    plt.title('s1 data set')
    plt.scatter(data[:,0], data[:,1], color='black', s=1)
    plt.show()

if __name__ == '__main__':
    main()