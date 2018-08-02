#! /usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kalman_filter import KalmanFilter


def plot2D(xMeas, yMeas, xKalman, yKalman):
    plt.plot(xMeas, yMeas, 'ro')
    plt.plot(xKalman, yKalman, 'g-')


def plot3D(xMeas, yMeas, zMeas, xKalman, yKalman, zKalman, N):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.scatter(xKalman, yKalman, zKalman, np.linspace(0,1,N), color='green')
    ax.scatter(xMeas, yMeas, zMeas, np.linspace(0,1,N), color='red')


def setUp(N):
    print "Setting up demo measurements"

    xReal = np.linspace(0.0, 10.0, N)
    yReal = xReal**2
    zReal = xReal

    xMeas = xReal + (np.random.random(N)-0.5)
    yMeas = yReal + (np.random.random(N)-0.5)
    zMeas = zReal + (np.random.random(N)-0.5)

    #plot 'real' measurements
    #plt.plot(xReal, yReal, 'b-')

    return zip(xMeas, yMeas, zMeas)

def calculateKalman(measurements):
    kf = KalmanFilter(dt=1, r = 0.6, q =0.001, xvals = 3, ndims = 3)
    results = []

    print "Calculating..."
    for measurement in measurements:
        measArray = np.asarray([[measurement[0]],[measurement[1]],[measurement[2]]])
        x = kf.run(measArray)
        results.append((x[:3]).tolist())

    return results

def runDemo():
    N = 60
    measurements = setUp(N)

    results = calculateKalman(measurements)

    xMeas, yMeas, zMeas = zip(*measurements)
    xKalman, yKalman, zKalman = zip(*results)

    #plot2D(xMeas, yMeas, xKalman, yKalman)
    plot3D(xMeas, yMeas, zMeas, xKalman, yKalman, zKalman, N)
    
    plt.show()

if __name__ == '__main__':
    runDemo()
