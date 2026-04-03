"""
Bu script YALNIZCA TensorFlow kullanir (OR-Tools yoktur).
route_optimizer.py tarafindan subprocess olarak cagrilir,
filtreli konteynerleri JSON formatinda stdout'a yazar.
"""
import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def predict_and_filter(veri_yolu, model_yolu, threshold):
    df = pd.read_excel(veri_yolu)
    model = tf.keras.models.load_model(model_yolu)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['olcekli_doluluk'] = scaler.fit_transform(df[['doluluk_sayisal']])

    hedef_konteynerler = []

    for konteyner_id, grup in df.groupby('konteyner_id'):
        grup = grup.sort_values(by=['tarih', 'saat'])
        son_5_veri = grup['olcekli_doluluk'].values[-5:]

        if len(son_5_veri) < 5:
            continue

        X_tahmin = np.reshape(son_5_veri, (1, 5, 1))
        tahmin_olcekli = model.predict(X_tahmin, verbose=0)
        tahmin_gercek = float(scaler.inverse_transform(tahmin_olcekli)[0][0])

        if tahmin_gercek >= threshold:
            hedef_konteynerler.append({
                'id': str(konteyner_id),
                'enlem': float(grup['enlem'].iloc[-1]),
                'boylam': float(grup['boylam'].iloc[-1]),
                'tahmin_doluluk': round(tahmin_gercek, 2)
            })

    return hedef_konteynerler


if __name__ == "__main__":
    veri_yolu = sys.argv[1]
    model_yolu = sys.argv[2]
    threshold = float(sys.argv[3])

    sonuc = predict_and_filter(veri_yolu, model_yolu, threshold)
    # Sadece JSON yaz - route_optimizer okuyacak
    print(json.dumps(sonuc))
