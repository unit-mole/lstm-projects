
import numpy as np
from src.data_preprocessing import generate_har_data, validate_window

def test_generator_shape():
 X,y,n=generate_har_data(12,80,6,42); assert X.shape==(12,80,6); assert y.shape==(12,); assert len(n)==6

def test_determinism():
 a,_,_=generate_har_data(3,seed=7); b,_,_=generate_har_data(3,seed=7); assert np.allclose(a,b)

def test_validation():
 assert validate_window(np.zeros((80,6))).shape==(80,6)
