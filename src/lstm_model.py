import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_preprocessing import load_and_clean_data, prepare_lstm_data


def build_and_train_lstm(X_train, y_train, epochs=10, batch_size=16):
    print("\n--- LSTM Modeli İnşa Ediliyor ---")
    model = Sequential()

    # 1. LSTM Katmani
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # 2. LSTM Katmani
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Cikis Katmani
    model.add(Dense(units=1))

    # Modeli Derle
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Model eğitimi başlıyor. Lütfen bekleyin...")
    # Egitimi Baslat (verinin %20'si dogrulama icin ayrildi)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')
    temiz_veri = load_and_clean_data(veri_yolu)
    X, y, scaler, islenmis_veri = prepare_lstm_data(temiz_veri, window_size=5)

    # Modeli Egit (Deneme amacli 5 epoch)
    egitilmis_model = build_and_train_lstm(X, y, epochs=5, batch_size=16)

    # Egitilen Modeli Kaydet
    model_kayit_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')
    egitilmis_model.save(model_kayit_yolu)

    print(f"\nModel başarıyla eğitildi ve kaydedildi: {model_kayit_yolu} ✅")