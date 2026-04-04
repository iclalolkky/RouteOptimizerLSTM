import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

EXPECTED_COLUMNS = [
    'konteyner_id', 'tarih', 'saat', 'gun', 'enlem', 'boylam',
    'doluluk_orani', 'doluluk_sayisal', 'harita_linki'
]


def load_and_clean_data(file_path):
    print("Veri seti yükleniyor...")
    df = pd.read_excel(file_path)

    if not set(EXPECTED_COLUMNS).issubset(df.columns):
        if len(df.columns) == len(EXPECTED_COLUMNS):
            df = pd.read_excel(file_path, header=None, names=EXPECTED_COLUMNS)
        else:
            raise ValueError(
                "Veri seti beklenen kolon yapısına sahip değil. "
                f"Bulunan kolonlar: {list(df.columns)}"
            )

    df['tarih'] = df['tarih'].astype(str)
    df['saat'] = df['saat'].astype(str)
    df['enlem'] = pd.to_numeric(df['enlem'], errors='coerce')
    df['boylam'] = pd.to_numeric(df['boylam'], errors='coerce')
    df['doluluk_sayisal'] = pd.to_numeric(df['doluluk_sayisal'], errors='coerce')
    df['tarih_saat'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'], errors='coerce')

    df = df.dropna(subset=['konteyner_id', 'tarih_saat', 'enlem', 'boylam', 'doluluk_sayisal']).copy()

    columns_to_drop = [
        column for column in ['tarih', 'saat', 'gun', 'doluluk_orani', 'harita_linki']
        if column in df.columns
    ]
    df = df.drop(columns=columns_to_drop)

    df = df.sort_values(by=['konteyner_id', 'tarih_saat']).reset_index(drop=True)

    print("Veri temizleme tamamlandı.")
    return df


def prepare_lstm_data(df, window_size=5):
    print(f"LSTM için zaman pencereleri (window_size={window_size}) oluşturuluyor...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    df = df.copy()
    df['olcekli_doluluk'] = scaler.fit_transform(df[['doluluk_sayisal']])

    X, y = [], []
    konteyner_gruplari = df.groupby('konteyner_id')

    for _, grup in konteyner_gruplari:
        degerler = grup['olcekli_doluluk'].values
        for i in range(len(degerler) - window_size):
            X.append(degerler[i:(i + window_size)])
            y.append(degerler[i + window_size])

    print(f"Toplam oluşturulan veri dizisi (sequence) sayısı: {len(X)}")

    if not X:
        raise ValueError(
            f"window_size={window_size} için yeterli eğitim verisi oluşmadı. "
            "Her konteyner için en az 6 zaman noktası gerekli."
        )

    X_dizisi = np.array(X)
    X_dizisi = np.reshape(X_dizisi, (X_dizisi.shape[0], X_dizisi.shape[1], 1))

    return X_dizisi, np.array(y), scaler, df


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'cop_veri_seti.xlsx')

    try:
        temiz_veri = load_and_clean_data(veri_yolu)
        X, y, scaler, islenmis_veri = prepare_lstm_data(temiz_veri, window_size=5)

        print("\n--- İlk Veri Dizisi Örneği (X) ---")
        print(X[0])
        print(f"Buna Karşılık Gelen Hedef Değer (y): {y[0]}")

        print("\nVeri ön işleme adımı tamamlandı!")

    except FileNotFoundError:
        print(f"\nHATA: '{veri_yolu}' yolunda veri dosyası bulunamadı.")
        print("Lütfen veri setinin adının 'cop_veri_seti.xlsx' olduğundan ve 'data' klasörünün içinde olduğundan emin ol.")
    except ValueError as error:
        print(f"\nHATA: {error}")