import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
import requests


def get_osrm_distance_matrix(konteynerler):
    print("\nOSRM üzerinden yol mesafeleri hesaplanıyor...")
    coords = ";".join([f"{k['boylam']},{k['enlem']}" for k in konteynerler])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            matrix = np.array(data['distances'], dtype=int)
            print("Yol verisi başarıyla çekildi!")
            return matrix
        else:
            print(f"OSRM API Hatası (Kod: {response.status_code}).")
            return None
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return None


def get_predictions_and_filter(veri_yolu, model_yolu, threshold=28):
    print("Veriler ve Model yükleniyor...")
    df = pd.read_excel(veri_yolu)
    model = tf.keras.models.load_model(model_yolu)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['olcekli_doluluk'] = scaler.fit_transform(df[['doluluk_sayisal']])

    hedef_konteynerler = []

    konteyner_gruplari = df.groupby('konteyner_id')
    for konteyner_id, grup in konteyner_gruplari:
        grup = grup.sort_values(by=['tarih', 'saat'])
        son_5_veri = grup['olcekli_doluluk'].values[-5:]

        if len(son_5_veri) < 5:
            continue

        X_tahmin = np.reshape(son_5_veri, (1, 5, 1))
        tahmin_olcekli = model.predict(X_tahmin, verbose=0)
        tahmin_gercek = scaler.inverse_transform(tahmin_olcekli)[0][0]

        if tahmin_gercek >= threshold:
            enlem = grup['enlem'].iloc[-1]
            boylam = grup['boylam'].iloc[-1]
            hedef_konteynerler.append({
                'id': konteyner_id,
                'enlem': enlem,
                'boylam': boylam,
                'tahmin_doluluk': round(tahmin_gercek, 2)
            })

    print(f"\nTahmin tamamlandı! %{threshold} üzeri doluluğa ulaşacak {len(hedef_konteynerler)} konteyner bulundu.")
    return hedef_konteynerler


def create_route(konteynerler):
    if len(konteynerler) < 2:
        print("Rota oluşturmak için yeterli sayıda dolu konteyner yok.")
        return

    print("\nOR-Tools ile Rota Optimizasyonu Başlıyor...")
    num_locations = len(konteynerler)

    distance_matrix = get_osrm_distance_matrix(konteynerler)

    if distance_matrix is None:
        print(
            "HATA: Yol verisi alınamadığı için rota optimizasyonu iptal edildi. Lütfen internet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.")
        return

    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print("\n--- ÇÖP TOPLAMA ROTASI ---")
        index = routing.Start(0)
        rota_sirasi = []
        toplam_mesafe = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            rota_sirasi.append(konteynerler[node_index])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            toplam_mesafe += routing.GetArcCostForVehicle(previous_index, index, 0)

        print("\nHarita oluşturuluyor...")
        baslangic_enlem = rota_sirasi[0]['enlem']
        baslangic_boylam = rota_sirasi[0]['boylam']
        harita = folium.Map(location=[baslangic_enlem, baslangic_boylam], zoom_start=15)

        koordinatlar_listesi = []
        for i, nokta in enumerate(rota_sirasi):
            print(f"{i + 1}. Durak: Konteyner {nokta['id']} (Doluluk: %{nokta['tahmin_doluluk']})")
            koordinatlar_listesi.append([nokta['enlem'], nokta['boylam']])

            if i == 0:
                renk, ikon = 'green', 'play'
            elif i == len(rota_sirasi) - 1:
                renk, ikon = 'red', 'stop'
            else:
                renk, ikon = 'blue', 'trash'

            folium.Marker(
                location=[nokta['enlem'], nokta['boylam']],
                popup=f"<b>{i + 1}. Durak</b><br>Konteyner: {nokta['id']}<br>Doluluk: %{nokta['tahmin_doluluk']}",
                icon=folium.Icon(color=renk, icon=ikon)
            ).add_to(harita)

        folium.PolyLine(locations=koordinatlar_listesi, color='red', weight=4, opacity=0.8).add_to(harita)

        print(f"\nToplam Katedilecek Yol Mesafesi: {toplam_mesafe} metre")

        harita_kayit_yolu = os.path.join(os.path.dirname(__file__), '..', 'optimize_rota_haritasi.html')
        harita.save(harita_kayit_yolu)

        mutlak_yol = os.path.abspath(harita_kayit_yolu).replace(os.sep, '/')
        print(f" Oluşturulan rotayı haritada inceleyebilmek için bu linke tıklayın:\n   file:///{mutlak_yol}")

    else:
        print("Uygun bir rota bulunamadı!")


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi.xlsx')
    model_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')

    filtreli_konteynerler = get_predictions_and_filter(veri_yolu, model_yolu, threshold=28)
    create_route(filtreli_konteynerler)