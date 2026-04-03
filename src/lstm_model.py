import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Kendi yazdığın veri işleme modülünü içe aktarıyoruz
from data_preprocessing import load_and_clean_data, prepare_lstm_data


def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16):
    print("\n--- LSTM Modeli İnşa Ediliyor ---")
    model = Sequential()

    # 1. LSTM Katmanı: Verideki zaman serisi örüntülerini yakalar
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Ezberlemeyi önlemek için %20 nöron rastgele kapatılır

    # 2. LSTM Katmanı
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    # Çıkış Katmanı: Tek bir doluluk oranı tahmini (sayısal değer)
    model.add(Dense(units=1))

    # Modeli Derleme (Optimizer ve Kayıp Fonksiyonu)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # --- GELİŞMİŞ EĞİTİM KONTROLLERİ (PRO DOKUNUŞU) ---
    # 1. EarlyStopping: Model 7 epoch boyunca hiç gelişmezse, 50'yi beklemeden eğitimi keser.
    # restore_best_weights=True sayesinde en iyi performansı gösterdiği ana geri döner.
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # 2. ReduceLROnPlateau: Model öğrenmede tıkanırsa, öğrenme hızını (learning rate) yarıya düşürür.
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

    print(f"Model eğitimi maksimum {epochs} epoch üzerinden başlıyor...")

    # Eğitimi Başlat (Verinin %20'si doğrulama/test için ayrıldı)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, lr_reducer],
        verbose=1  # Eğitim sürecini konsolda adım adım gösterir
    )

    return model


if __name__ == "__main__":
    print("Sistem başlatılıyor, yollar ayarlanıyor...")

    # Dosya yollarını dinamik olarak ayarla (Klasör yapısına uygun)
    base_dir = os.path.dirname(__file__)
    veri_yolu = os.path.join(base_dir, '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')

    # Modellerin kaydedileceği klasörün var olduğundan emin ol
    models_dir = os.path.join(base_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_kayit_yolu = os.path.join(models_dir, 'lstm_doluluk_modeli.keras')
    scaler_kayit_yolu = os.path.join(models_dir, 'scaler.pkl')

    # 1. Veriyi yükle ve temizle
    print(f"Veri okunuyor: {veri_yolu}")
    temiz_veri = load_and_clean_data(veri_yolu)

    # 2. LSTM için veriyi hazırla (Scaler burada üretilir ve veri 0-1 arasına sıkıştırılır)
    X, y, scaler, islenmis_veri = prepare_lstm_data(temiz_veri, window_size=5)

    # 3. Modeli Eğit
    egitilmis_model = build_and_train_lstm(X, y, epochs=50, batch_size=16)

    # 4. Eğitilen Modeli Kaydet
    egitilmis_model.save(model_kayit_yolu)

    # 5. Scaler'ı Kaydet (Tahmin kodunda kullanmak için ÇOK ÖNEMLİ)
    with open(scaler_kayit_yolu, 'wb') as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 50)
    print("✅ EĞİTİM BAŞARIYLA TAMAMLANDI!")
    print(f"📍 Eğitilen Model : {model_kayit_yolu}")
    print(f"📍 Scaler Dosyası : {scaler_kayit_yolu}")
    print("=" * 50)