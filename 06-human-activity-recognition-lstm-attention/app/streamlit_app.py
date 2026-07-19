
from pathlib import Path
import sys, json
import numpy as np, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_preprocessing import generate_har_data, FEATURE_NAMES
from src.inference_pipeline import HARInferencePipeline
st.set_page_config(page_title='HAR with LSTM Attention',page_icon='🏃',layout='wide')
@st.cache_resource
def load_pipe(): return HARInferencePipeline(ROOT/'models/lstm_attention_har.keras',ROOT/'models/har_meta.json')
pipe=load_pipe()
st.title('Human Activity Recognition using LSTM with Attention')
st.write('Classify synthetic accelerometer and gyroscope sequences using a pretrained temporal-attention model.')
st.warning('Educational portfolio demo only. Do not use for healthcare, safety, surveillance, insurance, employment, or legal decisions. Do not upload private sensor data.')
mode=st.radio('Choose input method',['Generated sample','Upload CSV'],horizontal=True)
window=None; source=''
if mode=='Generated sample':
    activity=st.selectbox('Select an activity',pipe.class_names)
    seed=st.number_input('Sample seed',0,100000,42)
    X,y,names=generate_har_data(n_samples=500,seed=int(seed)); candidates=np.where(y==names.index(activity))[0]; window=X[int(candidates[0])]; source=f'Generated {activity} sample'
else:
    up=st.file_uploader('Upload an 80-row CSV with six sensor columns',type=['csv'])
    if up:
        df=pd.read_csv(up); missing=[c for c in FEATURE_NAMES if c not in df.columns]
        if missing: st.error('Missing columns: '+', '.join(missing))
        else: window=df[FEATURE_NAMES].to_numpy(dtype='float32'); source=up.name
if window is not None:
    df=pd.DataFrame(window,columns=FEATURE_NAMES); st.subheader('Sensor window'); st.line_chart(df)
    if st.button('Predict activity',type='primary'):
        try:
            r=pipe.predict(window); c1,c2=st.columns(2); c1.metric('Predicted activity',r['predicted_activity'].replace('_',' ').title()); c2.metric('Confidence',f"{r['confidence']:.1%}")
            probs=pd.DataFrame({'Activity':[x.replace('_',' ').title() for x in r['probabilities']], 'Probability':list(r['probabilities'].values())}).set_index('Activity')
            st.subheader('Activity probabilities'); st.bar_chart(probs)
            st.subheader('Top 3 predictions'); st.dataframe(pd.DataFrame([(a.replace('_',' ').title(),f'{p:.2%}') for a,p in r['top3']],columns=['Activity','Probability']),hide_index=True,use_container_width=True)
            out=pd.DataFrame([{'source':source,'predicted_activity':r['predicted_activity'],'confidence':r['confidence'],**{f'prob_{k}':v for k,v in r['probabilities'].items()}}])
            st.download_button('Download prediction CSV',out.to_csv(index=False),'har_prediction.csv','text/csv')
        except Exception as e: st.error(str(e))
with st.expander('Model details'):
 st.write('Two stacked LSTM layers (96 and 64 units), temporal attention, dense classification head, and six-way softmax output. Input shape: 80 time steps × 6 sensor features.')
with st.expander('Limitations'):
 st.write('The model was trained on deterministic synthetic signals. High benchmark accuracy does not imply real-world wearable-device performance. Sensor placement, sampling rate, device calibration, people, and environments can materially change results.')
