import os
import pickle

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from data_preprocessing import load_and_clean_data, prepare_lstm_data


def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16):
    print("\n--- LSTM Modeli ---")

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    ]

    print(f"Model eğitimi maksimum {epochs} epoch üzerinden başlıyor...")
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    return model


if __name__ == "__main__":
    print("Sistem başlatılıyor, yollar ayarlanıyor...")

    base_dir = os.path.dirname(__file__)
    veri_yolu = os.path.join(base_dir, '..', 'data', 'cop_veri_seti.xlsx')

    models_dir = os.path.join(base_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_kayit_yolu = os.path.join(models_dir, 'lstm_doluluk_modeli.keras')
    scaler_kayit_yolu = os.path.join(models_dir, 'doluluk_scaler.pkl')

    print(f"Veri okunuyor: {veri_yolu}")
    temiz_veri = load_and_clean_data(veri_yolu)

    X, y, scaler, islenmis_veri = prepare_lstm_data(temiz_veri, window_size=5)
    egitilmis_model = build_and_train_lstm(X, y, epochs=50, batch_size=16)

    egitilmis_model.save(model_kayit_yolu)

    with open(scaler_kayit_yolu, 'wb') as scaler_dosyasi:
        pickle.dump(scaler, scaler_dosyasi)

    print("\n" + "=" * 50)
    print(" EĞİTİM TAMAMLANDI!")
    print(f" Eğitilen Model : {model_kayit_yolu}")
    print(f" Scaler Dosyası : {scaler_kayit_yolu}")
    print("=" * 50)
