import os
import sys
import json
import logging
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import requests
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def get_predictions_and_filter(veri_yolu, model_yolu, threshold=45):
    """
    LSTM tahminini ayri bir subprocess'te (predict_containers.py) calistirir.
    TensorFlow ve OR-Tools ayni process'te calisinca protobuf catismasi
    nedeniyle program cokuyor; bu mimari bunu onler.
    """
    print("Sistem Başlatılıyor: LSTM tahmini subprocess'te çalıştırılıyor...")

    predict_script = os.path.join(os.path.dirname(__file__), 'predict_containers.py')
    result = subprocess.run(
        [sys.executable, predict_script, veri_yolu, model_yolu, str(threshold)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"HATA: Tahmin scripti başarısız oldu:\n{result.stderr}")
        return []

    hedef_konteynerler = json.loads(result.stdout)
    print(f"Tahmin tamamlandı! %{threshold} üzeri doluluğa ulaşacak "
          f"toplam {len(hedef_konteynerler)} konteyner bulundu.")
    return hedef_konteynerler


def get_osrm_distance_matrix(konteynerler):
    coords = ";".join([f"{k['boylam']},{k['enlem']}" for k in konteynerler])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            matrix = np.array(data['distances'], dtype=int)
            return matrix
        else:
            print(f"OSRM API Hatası (Kod: {response.status_code}).")
            return None
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return None


def vardiyalara_bol(konteynerler):
    if len(konteynerler) < 2:
        return konteynerler, []

    # Centroid depot: her iki vardiyada ayni merkezi baslangic noktasi
    centroid_enlem = sum(k['enlem'] for k in konteynerler) / len(konteynerler)
    centroid_boylam = sum(k['boylam'] for k in konteynerler) / len(konteynerler)
    depo = {'id': 'DEPO', 'enlem': centroid_enlem, 'boylam': centroid_boylam, 'tahmin_doluluk': 0}

    # Yol agi bazli clustering: OSRM mesafe matrisi + MDS ile 2D projeksiyon
    # Cografik koordinatlarla K-Means, yollarla birbirinden kopuk olan
    # konteynerleri (nehir, tek yon vb. engellerle ayrilmis) yanlis eslestiriyor.
    # MDS, yol mesafelerini 2D uzaya donusturur; K-Means bu uzayda calisir.
    print("Yol ağı bazlı kümeleme için mesafe matrisi hesaplanıyor...")
    distance_matrix = get_osrm_distance_matrix(konteynerler)

    if distance_matrix is not None:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
        # Simetrik matris yap (OSRM asimetrik olabilir)
        sym_matrix = ((distance_matrix + distance_matrix.T) / 2).astype(float)
        koordinatlar_2d = mds.fit_transform(sym_matrix)
    else:
        print("[!] OSRM alinamadi, cografik koordinat kullaniliyor.")
        koordinatlar_2d = np.array([[k['enlem'], k['boylam']] for k in konteynerler])

    # MDS'in ilk boyutuna gore sirala ve ortadan ikiye esit bol.
    n = len(konteynerler)
    yarim = n // 2
    siralama = np.argsort(koordinatlar_2d[:, 0])
    sabah_idx = set(siralama[:yarim])
    aksam_idx = set(siralama[yarim:])

    # --- IZOLE KONTEYNER DUZELTMESI ---
    # Yol agiyla birbirinden kopuk konteynerlerin hepsi ayni vardiyada olmalidir.
    # Ortalama mesafe karsilastirmasi yaniltici: her iki vardiyada da ana alan
    # konteynerleri var, bu yuzden K054 gibi izole bir konteyner her iki gruba
    # da esit uzak gorunuyor.
    #
    # Duzeltme: EN YAKIN KOMSU karsilastirmasi kullan.
    # K054 -> sabah en yakini K053 = ~100m, aksam en yakini = ~3500m.
    # Diger vardiyadaki en yakin uye 5x daha yakinsa konteyner yanlis tarafafta.
    if distance_matrix is not None:
        degisti = True
        while degisti:
            degisti = False
            for idx_set, diger_set in [(sabah_idx, aksam_idx), (aksam_idx, sabah_idx)]:
                for c_idx in list(idx_set):
                    kendi_mesafeler = [distance_matrix[c_idx][j] for j in idx_set if j != c_idx]
                    diger_mesafeler = [distance_matrix[c_idx][j] for j in diger_set]
                    if not kendi_mesafeler or not diger_mesafeler:
                        continue
                    en_yakin_kendi = min(kendi_mesafeler)
                    en_yakin_diger = min(diger_mesafeler)
                    # Diger vardiyadaki en yakin uye 5x daha yakinsa tasi
                    if en_yakin_diger * 5 < en_yakin_kendi:
                        idx_set.remove(c_idx)
                        diger_set.add(c_idx)
                        degisti = True

    sabah_vardiyasi = [depo] + [konteynerler[i] for i in sabah_idx]
    aksam_vardiyasi = [depo] + [konteynerler[i] for i in aksam_idx]

    return sabah_vardiyasi, aksam_vardiyasi


def create_route(konteynerler, vardiya_adi):
    if len(konteynerler) < 6:
        print(f"\n[!] {vardiya_adi} için yeterli sayıda dolu konteyner yok "
              f"({len(konteynerler)-1} adet, en az 5 gerekli).")
        return

    print(f"\n{'=' * 60}")
    print(f" {vardiya_adi.upper()} ROTA RAPORU (5 KAMYON)")
    print(f"{'=' * 60}")
    print(f"Toplam konteyner: {len(konteynerler)-1}")
    print("OSRM üzerinden karayolu mesafeleri hesaplanıyor...")

    num_locations = len(konteynerler)
    num_vehicles = 5
    depot = 0

    distance_matrix = get_osrm_distance_matrix(konteynerler)

    if distance_matrix is None:
        print("HATA: Karayolu verisi alınamadı. Rota iptal edildi.")
        return

    print("Yol verisi başarıyla çekildi. Rota Optimizasyonu başlatılıyor (15 sn sürebilir)...")

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- KM DENGESI ---
    # SetGlobalSpanCostCoefficient: en uzun - en kisa rota farkini minimize eder.
    # Mesafeler kucuk oldugunda (bu projede ~1.5km alan) maliyet fonksiyonunda
    # agirlik vermek yeterli; hard constraint degil soft constraint.
    max_arac_mesafesi = 100_000  # kamyon basi max 100 km
    routing.AddDimension(
        transit_callback_index,
        0,
        max_arac_mesafesi,
        True,
        'Distance'
    )
    distance_dimension = routing.GetDimensionOrDie('Distance')
    # GlobalSpanCostCoefficient: yol agi bircok alanda kucuk cografik alanlarda
    # dahi uzun yol mesafeleri olusturabilir. Bu durumda span katsayisi cok
    # yuksek yapilinca kisitlar catisiyor. Km dengesini durak sayisi kisiti
    # sagliyor; span katsayi sifir.
    distance_dimension.SetGlobalSpanCostCoefficient(0)

    # --- DURAK SAYISI DENGESI ---
    # Her kamyona tam olarak ceil(N/5) veya 1 az durak verilir.
    # PATH_CHEAPEST_ARC en yakin konteyneri sectiginden, yol agiyla
    # birbirinden kopuk konteynerler (K051-K054 gibi) dogal olarak
    # ayni kamyona dusuyor - bu uzun rota sorununu minimize eder.
    stop_demands = [0] + [1] * (num_locations - 1)
    stops_per_vehicle = math.ceil((num_locations - 1) / num_vehicles)
    max_stops = stops_per_vehicle  # tampon yok: esit dagilimi zorla

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return stop_demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimension(
        demand_callback_index,
        0,
        max_stops,
        True,
        'Capacity'
    )

    # Doluluk (agirlik) bilgisi yalnizca raporlama icin
    doluluk_demands = [0] + [max(1, int(k['tahmin_doluluk'])) for k in konteynerler[1:]]

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # PATH_CHEAPEST_ARC: en yakin komsu secimi - yol agi izole gruplarini
    # dogal olarak tek bir kamyona toplar.
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        toplam_filo_mesafesi = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            rota_guzergahi = []
            arac_mesafesi = 0
            arac_yuzdoluluk = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index == 0:
                    rota_guzergahi.append("DEPO")
                else:
                    rota_guzergahi.append(konteynerler[node_index]['id'])
                    arac_yuzdoluluk += doluluk_demands[node_index]

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                arac_mesafesi += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

            rota_guzergahi.append("DEPO")
            toplam_filo_mesafesi += arac_mesafesi

            if len(rota_guzergahi) > 2:
                arac_km = round(arac_mesafesi / 1000, 2)
                print(f"\n  Kamyon {vehicle_id + 1}:")
                print(f"    Konteyner Sayisi : {len(rota_guzergahi) - 2} adet")
                print(f"    Toplam Doluluk   : {arac_yuzdoluluk} birim")
                print(f"    Katedilen Mesafe : {arac_km} km")
                print(f"    Guzergah         : {' -> '.join(rota_guzergahi)}")

        toplam_km = round(toplam_filo_mesafesi / 1000, 2)
        print(f"\n  Toplam Filo Mesafesi: {toplam_km} km")

    else:
        print("Uygun bir rota bulunamadı! (Matematiksel olarak çözülemedi)")


if __name__ == "__main__":
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'bosna_hersek_cop_verisi_gercekci.xlsx')
    model_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')

    filtreli_konteynerler = get_predictions_and_filter(veri_yolu, model_yolu, threshold=45)

    if not filtreli_konteynerler:
        print("Hiç konteyner bulunamadı, program sonlandı.")
        sys.exit(1)

    sabah_listesi, aksam_listesi = vardiyalara_bol(filtreli_konteynerler)

    print(f"\nVardiya dağılımı:")
    print(f"  Sabah: {len(sabah_listesi)-1} konteyner")
    print(f"  Akşam: {len(aksam_listesi)-1} konteyner")

    create_route(sabah_listesi, "Sabah Vardiyası")
    create_route(aksam_listesi, "Akşam Vardiyası")
