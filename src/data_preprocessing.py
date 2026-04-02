import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def load_and_clean_data(file_path):
    print("Veri seti yükleniyor...")
    # Veriyi oku
    df = pd.read_excel(file_path)

    # Tarih ve saat sutunlarını birlestirip tek bir datetime yap
    df['tarih_saat'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'])

    # Gereksiz sutunları at
    columns_to_drop = ['tarih', 'saat', 'gun', 'doluluk_orani', 'harita_linki']
    df = df.drop(columns=columns_to_drop)

    # Veriyi once konteyner_id'ye, sonra zamana göre sirala
    df = df.sort_values(by=['konteyner_id', 'tarih_saat']).reset_index(drop=True)

    print("Veri temizleme tamamlandı.")
    return df

def prepare_lstm_data(df, window_size=5):
    print(f"LSTM için zaman pencereleri (window_size={window_size}) oluşturuluyor...")

    # 0 ile 1 arasi degerler LSTM'yi daha hizli ve kararli yapar
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Sadece doluluk_sayisal sutununu olceklendir (Normalizasyon)
    # Sonradan gercek yuzdeliklere cevirebilmek icin scaler'i sakla
    df['olcekli_doluluk'] = scaler.fit_transform(df[['doluluk_sayisal']])

    X, y = [], []

    # Her konteyner icin pencereleme (sliding window) yap
    konteyner_gruplari = df.groupby('konteyner_id')

    for konteyner_id, grup in konteyner_gruplari:
        degerler = grup['olcekli_doluluk'].values

        # window_size kadar gecmis veriyi alıp (X), bir sonraki adimi (y) hedef olarak belirle
        for i in range(len(degerler) - window_size):
            X.append(degerler[i:(i + window_size)])
            y.append(degerler[i + window_size])

    print(f"Toplam oluşturulan veri dizisi (sequence) sayısı: {len(X)}")

    # X'i 3 boyutlu hale getiriyoruz.
    X_dizisi = np.array(X)
    X_dizisi = np.reshape(X_dizisi, (X_dizisi.shape[0], X_dizisi.shape[1], 1))

    return X_dizisi, np.array(y), scaler, df


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')

    try:
        temiz_veri = load_and_clean_data(veri_yolu)
        X, y, scaler, islenmis_veri = prepare_lstm_data(temiz_veri, window_size=5)

        print("\n--- İlk Veri Dizisi Örneği (X) ---")
        print(X[0])
        print(f"Buna Karşılık Gelen Hedef Değer (y): {y[0]}")

        print("\nVeri ön işleme adımı BAŞARIYLA tamamlandı! ✅")

    except FileNotFoundError:
        print(f"\nHATA: '{veri_yolu}' yolunda veri dosyası bulunamadı.")
        print("Lütfen veri setinin adının 'cop_verisi.csv' olduğundan ve 'data' klasörünün içinde olduğundan emin ol.")