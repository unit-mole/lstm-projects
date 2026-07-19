
from __future__ import annotations
import numpy as np

CLASS_NAMES = ["walking","walking_upstairs","walking_downstairs","sitting","standing","laying"]
FEATURE_NAMES = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

def generate_har_data(n_samples: int = 3600, seq_len: int = 80, n_features: int = 6, seed: int = 42):
    if n_features != 6:
        raise ValueError("The supplied model expects exactly six sensor features.")
    rng=np.random.default_rng(seed); X=[]; y=[]; t=np.linspace(0,2*np.pi,seq_len)
    for _ in range(n_samples):
        cls=int(rng.integers(0,6)); s=np.zeros((seq_len,n_features),dtype='float32')
        if cls==0:
            s[:,0]=np.sin(2*t)+rng.normal(0,.15,seq_len); s[:,1]=np.cos(2*t)+rng.normal(0,.15,seq_len)
        elif cls==1:
            s[:,0]=1.2*np.sin(2.5*t)+rng.normal(0,.18,seq_len); s[:,1]=.8*np.cos(2.5*t)+rng.normal(0,.18,seq_len)
        elif cls==2:
            s[:,0]=np.sin(2.5*t+.7)+rng.normal(0,.18,seq_len); s[:,1]=-.7*np.cos(2.5*t)+rng.normal(0,.18,seq_len)
        elif cls==3: s += rng.normal(0,.05,(seq_len,n_features))
        elif cls==4: s += rng.normal(.02,.04,(seq_len,n_features))
        else: s += rng.normal(-.02,.03,(seq_len,n_features))
        s[:,2]=.5*s[:,0]+rng.normal(0,.05,seq_len); s[:,3]=.5*s[:,1]+rng.normal(0,.05,seq_len)
        s[:,4]=rng.normal(0,.08,seq_len)+(.2 if cls<3 else -.05)
        s[:,5]=rng.normal(0,.08,seq_len)+(.15 if cls in (1,2) else 0)
        X.append(s); y.append(cls)
    return np.asarray(X,dtype='float32'), np.asarray(y,dtype='int64'), CLASS_NAMES

def validate_window(values: np.ndarray, seq_len: int=80, n_features: int=6) -> np.ndarray:
    arr=np.asarray(values,dtype='float32')
    if arr.shape != (seq_len,n_features):
        raise ValueError(f"Expected shape ({seq_len}, {n_features}); received {arr.shape}.")
    if not np.isfinite(arr).all(): raise ValueError("Sensor window contains missing or infinite values.")
    return arr
