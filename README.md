# kalman-filter
Python implementation of a kalman filter, with demo scripts in 2 and 3 dimensions. 

Also has automatic state transition matrix creation to handle 3 state dimensions (e.g. position, velocity, acceleration) or up to 3 spacial dimensions.


To run:

Create a KalmanFilter object with desired initialisation settings then continually update with measurement arrays by calling the run method. e.g.:


def calculateKalman(measurements):

    kf = KalmanFilter(dt=1, zdims=2, r = 1000, q =0.0000001, xvals = 3)
    
    results = []
    
    for measurement in measurements:
    
        measArray = np.asarray([[measurement[0]],[measurement[1]]])
        
        x = kf.run(measArray)
       
        results.append((x[:2]).tolist())
       
        
    return results
