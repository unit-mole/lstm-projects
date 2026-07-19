
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .attention_layer import AttentionLayer
from .data_preprocessing import validate_window

class HARInferencePipeline:
    def __init__(self, model_path, metadata_path):
        self.model_path=Path(model_path); self.metadata=json.loads(Path(metadata_path).read_text())
        self.class_names=self.metadata['class_names']; self.seq_len=int(self.metadata['seq_len']); self.n_features=int(self.metadata['n_features'])
        self.model=tf.keras.models.load_model(self.model_path,custom_objects={'AttentionLayer':AttentionLayer},compile=False)
    def predict(self, window):
        x=validate_window(window,self.seq_len,self.n_features)[None,...]
        p=np.asarray(self.model.predict(x,verbose=0)[0],dtype=float); idx=int(np.argmax(p))
        order=np.argsort(p)[::-1]
        return {'predicted_activity':self.class_names[idx],'confidence':float(p[idx]),'probabilities':{self.class_names[i]:float(p[i]) for i in range(len(p))},'top3':[(self.class_names[i],float(p[i])) for i in order[:3]]}
