import numpy as np


class KF:
    def __init__(self, initial_x, initial_v, acc_variance):
        # mean of state GRV
        self._x = np.array([initial_x, initial_v]) 
        self._acc_variance = acc_variance

        #Covariance
        self._P = np.eye(2) #_P assigns a private memory and doesnot erase P once function is passed
                            
    def predict(self, dt): #time evolution from one step to other and predict
        #returns Nothing --- modifies state
        #x = F * x
        #P = F * P * Ft + G * Gt * a
        F = np.array([[1, dt], [0,1]])
        new_x = F.dot(self._x)
        G = np.array([0.5 * dt**2,dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T)*self._acc_variance
        
        self._P  = new_P
        self._x = new_x
    
    def update(self, meas_value, meas_variance):
        # y = Z -Hx 
        # S = H P Ht + R
        # K = P Ht S ^-1
        # x = x + ky
        #P = (I - K H) *P

        H = np.array([1,0]).reshape((1,2))

    
        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_P = (np.eye(2) - K.dot(H)).dot(self._P)
        new_x = self._x + K.dot(y)

        self._P = new_P
        self._x = new_x
        
    
    @property
    def mean(self):
        return self._x

    @property
    def cov(self):
        return self._P
    
    @property
    def pos(self):
        return self._x[0]
    
    @property
    def vel(self):
        return self._x[1]
    
    
    
