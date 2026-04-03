import os
import logging
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import requests
import math

# TensorFlow uyarılarını gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def get_osrm_distance_matrix(konteynerler):
    """OSRM API üzerinden koordinatlar arası gerçek sürüş mesafelerini (metre) çeker."""
    coords = ";".join([f"{k['boylam']},{k['enlem']}" for k in konteynerler])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            matrix = np.array(data['distances'], dtype=int)
            return matrix
        else:
            print(f"[-] OSRM API Hatası (Kod: {response.status_code}).")
            return None
    except Exception as e:
        print(f"[-] Bağlantı hatası: {e}")
        return None


def get_predictions_and_filter(veri_yolu, model_yolu, scaler_yolu, threshold=40):
    """LSTM modelini ve Scaler'ı kullanarak doluluk tahmini yapar ve sınırı aşanları filtreler."""
    print("Sistem Başlatılıyor: Veriler, Model ve Scaler yükleniyor...")

    df = pd.read_excel(veri_yolu)
    model = tf.keras.models.load_model(model_yolu)

    # DİKKAT: Eğitilmiş Scaler yükleniyor (Yeniden fit EDİLMİYOR!)
    with open(scaler_yolu, 'rb') as f:
        scaler = pickle.load(f)

    # Sadece transform işlemi yapıyoruz
    df['olcekli_doluluk'] = scaler.transform(df[['doluluk_sayisal']])

    hedef_konteynerler = []
    konteyner_gruplari = df.groupby('konteyner_id')

    for konteyner_id, grup in konteyner_gruplari:
        grup = grup.sort_values(by=['tarih', 'saat'])
        son_5_veri = grup['olcekli_doluluk'].values[-5:]

        if len(son_5_veri) < 5:
            continue

        # Tahmin için veriyi boyutlandır: (1 örnek, 5 zaman adımı, 1 özellik)
        X_tahmin = np.reshape(son_5_veri, (1, 5, 1))
        tahmin_olcekli = model.predict(X_tahmin, verbose=0)

        # Çıkan sonucu gerçek doluluk yüzdesine (0-100) çevir
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

    print(
        f"✅ Tahmin tamamlandı! %{threshold} üzeri doluluğa ulaşacak toplam {len(hedef_konteynerler)} konteyner bulundu.")
    return hedef_konteynerler


def vardiyalara_bol(konteynerler):
    """Konteynerleri K-Means algoritması ile Sabah ve Akşam olmak üzere iki coğrafi kümeye böler."""
    if not konteynerler:
        return [], []

    depo = konteynerler[0]  # İlk elemanı sabit depo olarak kabul ediyoruz
    kalanlar = konteynerler[1:]

    if len(kalanlar) > 1:
        coords = np.array([[k['enlem'], k['boylam']] for k in kalanlar])
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kumeler = kmeans.fit_predict(coords)
    else:
        kumeler = [0] * len(kalanlar)

    sabah_vardiyasi = [depo]
    aksam_vardiyasi = [depo]

    for i, kume_id in enumerate(kumeler):
        if kume_id == 0:
            sabah_vardiyasi.append(kalanlar[i])
        else:
            aksam_vardiyasi.append(kalanlar[i])

    return sabah_vardiyasi, aksam_vardiyasi


def create_route(konteynerler, vardiya_adi):
    """Belirtilen vardiya için OR-Tools kullanarak optimum rotayı çizer."""
    num_locations = len(konteynerler)

    if num_locations < 2:
        print(f"\n[!] {vardiya_adi} için toplanacak yeterli konteyner yok. Rota iptal.")
        return

    print(f"\n{'=' * 60}")
    print(f" 🚛 {vardiya_adi.upper()} ROTA RAPORU")
    print(f"{'=' * 60}")
    print("OSRM üzerinden karayolu mesafeleri hesaplanıyor...")

    # Dinamik Araç Sayısı: Eğer toplanacak konteyner sayısı 5'ten azsa, araç sayısını düşür.
    num_vehicles = min(5, num_locations - 1)
    depot = 0

    distance_matrix = get_osrm_distance_matrix(konteynerler)

    if distance_matrix is None:
        print("[-] HATA: Karayolu verisi alınamadı. Rota iptal edildi.")
        return

    print("✅ Yol verisi başarıyla çekildi. Rota Optimizasyonu başlatılıyor...")

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # --- 1. MESAFE BOYUTU ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(
        transit_callback_index,
        0,  # Bekleme süresi yok
        100000,  # Maksimum mesafe sınırı
        True,  # Mesafeler sıfırdan başlar
        'Distance'
    )

    # --- 2. KAPASİTE BOYUTU ---
    # Araçların kapasitesini esnetiyoruz (+2). Böylece algoritma çözümsüzlüğe düşmez.
    max_kapasite = int(math.ceil((num_locations - 1) / num_vehicles)) + 2

    demands = [0] + [1] * (num_locations - 1)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimension(
        demand_callback_index,
        0,  # Kapasite esnemesi yok
        max_kapasite,  # Araç başına maksimum alınabilecek konteyner
        True,  # Kapasite sayımı sıfırdan başlar
        'Capacity'
    )

    # --- 3. ÇÖZÜM STRATEJİSİ ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        toplam_filo_mesafesi = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            rota_guzergahi = []
            arac_mesafesi = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index == 0:
                    rota_guzergahi.append("DEPO")
                else:
                    # Doluluk oranını da raporda gösterelim
                    k_id = konteynerler[node_index]['id']
                    k_doluluk = konteynerler[node_index]['tahmin_doluluk']
                    rota_guzergahi.append(f"{k_id}(%{k_doluluk})")

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                arac_mesafesi += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            rota_guzergahi.append("DEPO")
            toplam_filo_mesafesi += arac_mesafesi

            # Eğer kamyon depodan çıkıp en az 1 konteyner aldıysa raporla
            if len(rota_guzergahi) > 2:
                print(f"\n🟢 Kamyon {vehicle_id + 1} Detayları:")
                print(f"   Toplanan Konteyner : {len(rota_guzergahi) - 2} adet")
                print(f"   Katedilen Mesafe   : {arac_mesafesi} metre")
                print(f"   Güzergah           : {' -> '.join(rota_guzergahi)}")

        print(f"\n🏁 {vardiya_adi} Toplam Filo Mesafesi: {toplam_filo_mesafesi} metre")

    else:
        print("\n[-] Uygun bir rota bulunamadı! Kısıtlamalar çok dar veya mesafe verileri hatalı olabilir.")


if __name__ == "__main__":
    # Dosya Yolları
    base_dir = os.path.dirname(__file__)
    veri_yolu = os.path.join(base_dir, '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')
    model_yolu = os.path.join(base_dir, '..', 'models', 'lstm_doluluk_modeli.keras')
    scaler_yolu = os.path.join(base_dir, '..', 'models', 'scaler.pkl')  # Eğitimden gelen scaler!

    # 1. Tahminleri al ve eşiği aşanları belirle (threshold=40 işlemi)
    filtreli_konteynerler = get_predictions_and_filter(veri_yolu, model_yolu, scaler_yolu, threshold=40)

    # 2. Seçilen konteynerleri iki coğrafi vardiyaya böl
    sabah_listesi, aksam_listesi = vardiyalara_bol(filtreli_konteynerler)

    # 3. Kamyon rotalarını oluştur
    create_route(sabah_listesi, "Sabah Vardiyası")
    create_route(aksam_listesi, "Akşam Vardiyası")