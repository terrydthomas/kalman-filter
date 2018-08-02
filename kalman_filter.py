import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, dt=0.1, ndims = 2, xvals = 2, zdims = None, pu = 1000, A = np.array([None]), r=1, q=0.05):
        """Initialize variable used by Kalman Filter class
     
        Args:
            dt: time step
            pu: inital uncertainty value
            zdims: number of measurement dimensions
            A: state transition matrix
            ndims: number of dimensions
            xvals: number of state values being measured - e.g. position, velocity
            xdims: number of state dimensions - e.g. xdims = 4 for [x,y,vx,vy] 
                   = ndims * xvals
            r: value for observation noise
            q: value for process noise
        Return:
            None
        """

        self.dt = float(dt) #time step
        self.ddt = (self.dt**2)/2

        # initialise state matrices
        xdims = ndims * xvals
        xsize = (xdims,1)

        self.x = np.zeros(xsize) # previous state vector
       
        self.P = np.eye(xdims)*pu # covariance matrix

        self.A = A        

        if not A.any():
            """
            if state transition matrix A is not provided default
            versions are made for 1, 2 or 3 values in the state vectors 
            (i.e position, velocity, acceleration) ; for either 1 or 2
            dimensions (e.g. (p, v, a) / (px, py, vx, vy, ax, ay)) / (p,v) etc.

            might need to provide an A matrix if sensor fusion is needed 
            (multiple measurements (z values) corresponding to same state values
            or for more state values or dimensions
            """
            if xdims > 2:
                self.A = np.eye(xdims) + np.diag(tuple([self.dt] * (ndims)),(xdims-ndims))
                if xvals ==3:
                    self.A = self.A + np.diag(tuple([self.ddt]*(xdims-ndims)),ndims)
            elif xdims == 2:
                self.A = np.eye(xdims) + np.array([[0,dt],[0,0]])
            elif xdims == 1:
                self.A = np.eye(xdims)


        self.Q = np.eye(self.x.shape[0]) * q # process noise matrix
        self.B = np.eye(self.x.shape[0])
        self.u = np.zeros((self.x.shape[0],1))

        # initialise observation matrices
        if not zdims:
            zdims = ndims

        zsize = (zdims,1)
        self.z = np.zeros(zsize)

        self.H = np.zeros((zdims,xdims))

        for i in range(0,ndims):
            self.H[i][i] = 1

        self.R = np.eye(self.z.shape[0]) * r

        self.lastResult = np.zeros(xsize)


    def predict(self):
        """Predict state vector, u and variance of uncertainty (covariance), P.
                where,
                x: previous state estimate
                P: previous covariance matrix (k-1)
                A: state transition (nxn) matrix (k-1)
                Q: process noise covariance matrix
                B: input effect matrix
                u: control input
            Equations:
                X_{k} = A * x_{k-1} + B * u_{k}
                P_{k} = A * P_{k-1} * A.T + Q
                where,
                    A.T is F transpose
            Args:
                None
            Return:
                vector of predicted state estimate, X

            Save to member variables:
                vector of predicted state estimate, X
                vector of covariance, P
            """

        # predicted state estimate
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # predicted estimate covariance
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

        # save last predicted result
        self.lastResult = self.x

        return self.x


    def update(self, z):
        """Correct or update state vector u and variance of uncertainty P (covariance).
                where,
                x: predicted state vector u
                H: observation effect matrix
                z: vector of observations (measurement vector)
                P: predicted covariance matrix
                Q: process noise matrix (diagonal values are process covariances)
                R: observation noise matrix (diagonal values are sensor covariances)

                K: Kalman Gain
                C: measurement prediction covariance matrix
                V: measurement mean
            Equations:
                C = H * P_{k} * H.T + R
                K_{k} = P_{k} * H.T(C.Inv)


                V_{k} = z_{k} - H * x_{k}
                x_{k} = x_{k} + K_{k} * V_{k}
                P_{k} = P_{k} - K_{k} * (C * K.T)

                where,
                    A.T is A transpose
                    C.Inv is C inverse
            Args:
                z: vector of observations
            Return:
                predicted state vector u
        """
        self.z = z

        C = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(C)))
        V = self.z - np.dot(self.H, self.x)

        self.x = self.x + np.dot(K, V)

        self.P = self.P - np.dot(K, np.dot(C, K.T))

        self.lastResult = self.x

        return self.x
        

    def run(self,z):

        xPredicted = self.predict()

        xUpdated = self.update(z)

        return xUpdated
