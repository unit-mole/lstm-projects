
"""Reproducible training entry point based on the original notebook."""
import json, random
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .attention_layer import AttentionLayer
from .data_preprocessing import generate_har_data

def build_model(input_shape,n_classes):
    inp=tf.keras.Input(shape=input_shape); x=tf.keras.layers.LSTM(96,return_sequences=True)(inp); x=tf.keras.layers.Dropout(.3)(x)
    x=tf.keras.layers.LSTM(64,return_sequences=True)(x); x=AttentionLayer()(x); x=tf.keras.layers.Dense(64,activation='relu')(x); x=tf.keras.layers.Dropout(.3)(x)
    out=tf.keras.layers.Dense(n_classes,activation='softmax')(x)
    m=tf.keras.Model(inp,out); m.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']); return m

def main():
    seed=42; random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    X,y,names=generate_har_data(seed=seed); Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=seed); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,stratify=ytmp,random_state=seed)
    m=build_model(Xtr.shape[1:],len(names)); m.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=20,batch_size=64,callbacks=[EarlyStopping(patience=4,restore_best_weights=True),ReduceLROnPlateau(patience=2)],verbose=1)
    out=Path('models'); out.mkdir(exist_ok=True); m.save(out/'lstm_attention_har.keras'); (out/'har_meta.json').write_text(json.dumps({'seq_len':80,'n_features':6,'class_names':names,'seed':seed},indent=2))
if __name__=='__main__': main()
