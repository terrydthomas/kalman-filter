#! /usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kalman_filter import KalmanFilter


def plot2D(xMeas, yMeas, xKalman, yKalman):
    plt.plot(xMeas, yMeas, 'ro')
    plt.plot(xKalman, yKalman, 'g-')


def plot3D(xMeas, yMeas, xKalman, yKalman, N):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    
    ax.plot(xKalman, yKalman, np.linspace(0,1,N), color='green')
    ax.scatter(xMeas, yMeas, np.linspace(0,1,N), color='red')


def setUp(N):
    print "Setting up demo measurements"

    xReal = np.linspace(0.0, 10.0, N)
    yReal = xReal**2

    xMeas = xReal + (np.random.random(N)-0.5)
    yMeas = yReal + (np.random.random(N)-0.5)

    #plot 'real' measurement
    #plt.plot(xReal, yReal, 'b-')

    return zip(xMeas, yMeas)

def calculateKalman(measurements):
    kf = KalmanFilter(dt=1, zdims=2, r = 1000, q =0.0000001, xvals = 3)
    results = []
    print "Calculating..."
    for measurement in measurements:
        measArray = np.asarray([[measurement[0]],[measurement[1]]])
        x = kf.run(measArray)
        results.append((x[:2]).tolist())

    return results

def runDemo():
    N = 80
    measurements = setUp(N)

    results = calculateKalman(measurements)

    xMeas, yMeas = zip(*measurements)
    xKalman, yKalman = zip(*results)

    plot2D(xMeas, yMeas, xKalman, yKalman)
    #plot3D(xMeas, yMeas, xKalman, yKalman, N)
    
    plt.show()

if __name__ == '__main__':
    runDemo()
