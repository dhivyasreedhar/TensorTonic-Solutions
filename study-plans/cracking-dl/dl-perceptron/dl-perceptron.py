import numpy as np

def perceptron(X: list, y: list, lr: float = 0.1, epochs: int = 100) -> tuple:
    """
    Returns the trained weight list and bias.
    """
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    w=np.zeros(X.shape[1],dtype=np.float64)
    b=0.0
    for _ in range(epochs):
        for point,target in zip(X,y):
            pred=1 if point@w+b>=0.0 else 0
            error=target-pred
            w+=lr*error*point
            b+=lr*error
    return [round(float(v),4) for v in w],round(float(b),4)
    
