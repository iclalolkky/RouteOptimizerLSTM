
import logging
import math
import os
import pickle
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import requests

NUM_VEHICLES = 5
WINDOW_SIZE = 5
DISTANCE_TOLERANCE_RATIO = 0.10
OSRM_BASE_URL = 'https://router.project-osrm.org'
EXPECTED_COLUMNS = [
    'konteyner_id', 'tarih', 'saat', 'gun', 'enlem', 'boylam',
    'doluluk_orani', 'doluluk_sayisal', 'harita_linki'
]

class BasicMinMaxScaler:
    def __init__(self):
        self.min_value = 0.0
        self.max_value = 1.0

    def fit(self, values):
        array = np.asarray(values, dtype=float).reshape(-1)
        self.min_value = float(np.min(array))
        self.max_value = float(np.max(array))
        return self

    def transform(self, values):
        array = np.asarray(values, dtype=float)
        delta = self.max_value - self.min_value
        if delta == 0:
            return np.zeros_like(array)
        return (array - self.min_value) / delta

    def inverse_transform(self, values):
        array = np.asarray(values, dtype=float)
        delta = self.max_value - self.min_value
        if delta == 0:
            return np.full_like(array, self.min_value)
        return (array * delta) + self.min_value

def load_prediction_data(file_path):
    df = pd.read_excel(file_path)

    if not set(EXPECTED_COLUMNS).issubset(df.columns):
        if len(df.columns) == len(EXPECTED_COLUMNS):
            df = pd.read_excel(file_path, header=None, names=EXPECTED_COLUMNS)
        else:
            raise ValueError(
                'Veri seti beklenen kolon yapısına sahip değil. '
                f'Bulunan kolonlar: {list(df.columns)}'
            )

    df['tarih'] = df['tarih'].astype(str)
    df['saat'] = df['saat'].astype(str)
    df['enlem'] = pd.to_numeric(df['enlem'], errors='coerce')
    df['boylam'] = pd.to_numeric(df['boylam'], errors='coerce')
    df['doluluk_sayisal'] = pd.to_numeric(df['doluluk_sayisal'], errors='coerce')
    df['tarih_saat'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'], errors='coerce')

    return df.dropna(
        subset=['konteyner_id', 'tarih_saat', 'enlem', 'boylam', 'doluluk_sayisal']
    ).copy()

def slugify(text):
    ceviri = str.maketrans('çğıöşüÇĞİÖŞÜ ', 'cgiosuCGIOSU_')
    temiz = text.translate(ceviri).lower()
    return re.sub(r'[^a-z0-9_]+', '_', temiz).strip('_')

def get_osrm_distance_matrix(konteynerler):
    coords = ';'.join([f"{k['boylam']},{k['enlem']}" for k in konteynerler])
    approaches = ';'.join(['curb'] * len(konteynerler))  # OSRM sağdan yanaşma düzeltmesi
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coords}?annotations=distance&approaches={approaches}"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        matrix = np.array(data['distances'], dtype=float)

        if np.isnan(matrix).any():
            print('OSRM bazı noktalar arasında yol bulamadı.')
            return None

        return np.rint(matrix).astype(int)
    except Exception as error:
        print(f'Bağlantı hatası: {error}')
        return None

def load_prediction_scaler(scaler_yolu, df):
    if scaler_yolu and os.path.exists(scaler_yolu):
        with open(scaler_yolu, 'rb') as scaler_dosyasi:
            return pickle.load(scaler_dosyasi)

    scaler = BasicMinMaxScaler()
    scaler.fit(df[['doluluk_sayisal']])
    return scaler

def get_predictions_and_filter(veri_yolu, model_yolu, scaler_yolu=None, threshold=40):
    import tensorflow as tf

    print('Sistem Başlatılıyor: Veriler ve Model yükleniyor...')
    df = load_prediction_data(veri_yolu)
    model = tf.keras.models.load_model(model_yolu)
    scaler = load_prediction_scaler(scaler_yolu, df)

    df['olcekli_doluluk'] = scaler.transform(df[['doluluk_sayisal']])
    hedef_konteynerler = []

    for konteyner_id, grup in df.groupby('konteyner_id'):
        grup = grup.sort_values(by='tarih_saat')
        son_veriler = grup['olcekli_doluluk'].values[-WINDOW_SIZE:]

        if len(son_veriler) < WINDOW_SIZE:
            continue

        X_tahmin = np.reshape(son_veriler, (1, WINDOW_SIZE, 1))
        tahmin_olcekli = model.predict(X_tahmin, verbose=0)
        tahmin_gercek = float(scaler.inverse_transform(tahmin_olcekli)[0][0])

        if tahmin_gercek >= threshold:
            son_satir = grup.iloc[-1]
            hedef_konteynerler.append({
                'id': konteyner_id,
                'enlem': float(son_satir['enlem']),
                'boylam': float(son_satir['boylam']),
                'tahmin_doluluk': round(tahmin_gercek, 2)
            })

    hedef_konteynerler = sorted(
        hedef_konteynerler,
        key=lambda item: item['tahmin_doluluk'],
        reverse=True
    )

    print(
        f"Tahmin tamamlandı! %{threshold} üzeri doluluğa ulaşacak toplam "
        f"{len(hedef_konteynerler)} konteyner bulundu."
    )
    return hedef_konteynerler

def build_depot(konteynerler):
    if not konteynerler:
        return {'id': 'DEPO', 'enlem': 0.0, 'boylam': 0.0, 'tahmin_doluluk': 0.0}

    return {
        'id': 'DEPO',
        'enlem': float(np.median([item['enlem'] for item in konteynerler])),
        'boylam': float(np.median([item['boylam'] for item in konteynerler])),
        'tahmin_doluluk': 0.0
    }

def vardiyalara_bol(konteynerler):
    if not konteynerler:
        depo = build_depot([])
        return [depo], [depo]

    depo = build_depot(konteynerler)
    merkez_enlem = depo['enlem']
    merkez_boylam = depo['boylam']

    sirali = sorted(
        konteynerler,
        key=lambda item: (
            math.atan2(item['enlem'] - merkez_enlem, item['boylam'] - merkez_boylam),
            ((item['enlem'] - merkez_enlem) ** 2 + (item['boylam'] - merkez_boylam) ** 2)
        )
    )

    sabah_vardiyasi = []
    aksam_vardiyasi = []

    for index, konteyner in enumerate(sirali):
        if index % 2 == 0:
            sabah_vardiyasi.append(konteyner)
        else:
            aksam_vardiyasi.append(konteyner)

    return [depo] + sabah_vardiyasi, [depo] + aksam_vardiyasi

def build_target_sizes(stop_count, num_vehicles):
    temel = stop_count // num_vehicles
    ekstra = stop_count % num_vehicles
    return [temel + (1 if vehicle_id < ekstra else 0) for vehicle_id in range(num_vehicles)]

def get_workload(distance, count):
    return distance + (count * 1500)


def route_workload_gap(route_distances, route_counts):
    if not route_distances:
        return 0.0
    aktif_is_yukleri = [get_workload(d, c) for d, c in zip(route_distances, route_counts) if c > 0]
    if not aktif_is_yukleri:
        return 0.0
    ortalama = float(sum(aktif_is_yukleri)) / len(aktif_is_yukleri)
    if ortalama == 0:
        return 0.0
    max_mutlak = max(abs(yuk - ortalama) for yuk in aktif_is_yukleri)
    max_oransal = max(abs(yuk - ortalama) / ortalama for yuk in aktif_is_yukleri)
    return max_oransal + (max_mutlak / 50000)

def detect_outliers(distance_matrix, stop_indices):
    depot_distances = [distance_matrix[0][idx] for idx in stop_indices]
    if not depot_distances:
        return set()
    median_dist = float(np.median(depot_distances))
    threshold = median_dist * 2.0
    return {idx for idx in stop_indices if distance_matrix[0][idx] > threshold}

def split_into_equal_count_clusters(distance_matrix, num_vehicles=NUM_VEHICLES):
    stop_indices = list(range(1, len(distance_matrix)))
    clusters = [[] for _ in range(num_vehicles)]

    if not stop_indices:
        return clusters

    outlier_indices = detect_outliers(distance_matrix, stop_indices)
    normal_indices = [idx for idx in stop_indices if idx not in outlier_indices]

    target_sizes = build_target_sizes(len(normal_indices), num_vehicles)

    sorted_normals = sorted(normal_indices, key=lambda idx: distance_matrix[0][idx])
    n = len(sorted_normals)
    step = max(1, n // num_vehicles)
    seed_pool = [sorted_normals[min(i * step + step // 2, n - 1)] for i in range(num_vehicles)]

    for vehicle_id in range(min(num_vehicles, len(seed_pool))):
        if target_sizes[vehicle_id] > 0:
            clusters[vehicle_id].append(seed_pool[vehicle_id])

    seeded = {c[0] for c in clusters if c}

    for node_index in [idx for idx in sorted_normals if idx not in seeded]:
        uygun_araclar = [
            v for v in range(num_vehicles)
            if len(clusters[v]) < target_sizes[v]
        ]
        if not uygun_araclar:
            uygun_araclar = list(range(num_vehicles))

        best_vehicle = min(
            uygun_araclar,
            key=lambda v: (
                cluster_route_distance(distance_matrix, clusters[v] + [node_index]),
                len(clusters[v])
            )
        )
        clusters[best_vehicle].append(node_index)

    for outlier_idx in sorted(outlier_indices, key=lambda idx: distance_matrix[0][idx]):
        best_vehicle = min(
            range(num_vehicles),
            key=lambda v: cluster_route_distance(
                distance_matrix, clusters[v] + [outlier_idx]
            )
        )
        clusters[best_vehicle].append(outlier_idx)

    return clusters

def split_into_distance_balanced_clusters(distance_matrix, num_vehicles=NUM_VEHICLES):
    stop_indices = list(range(1, len(distance_matrix)))
    clusters = [[] for _ in range(num_vehicles)]

    if not stop_indices:
        return clusters

    outlier_indices = detect_outliers(distance_matrix, stop_indices)
    normal_indices = [idx for idx in stop_indices if idx not in outlier_indices]

    sorted_normals = sorted(normal_indices, key=lambda idx: distance_matrix[0][idx])
    n = len(sorted_normals)
    step = max(1, n // num_vehicles)
    seed_pool = [sorted_normals[min(i * step + step // 2, n - 1)] for i in range(num_vehicles)]

    for vehicle_id in range(min(num_vehicles, len(seed_pool))):
        clusters[vehicle_id].append(seed_pool[vehicle_id])

    seeded = {c[0] for c in clusters if c}

    for node_index in [idx for idx in sorted_normals if idx not in seeded]:
        best_vehicle = min(
            range(num_vehicles),
            key=lambda v: cluster_route_distance(distance_matrix, clusters[v] + [node_index])
        )
        clusters[best_vehicle].append(node_index)

    for outlier_idx in sorted(outlier_indices, key=lambda idx: distance_matrix[0][idx]):
        best_vehicle = min(
            range(num_vehicles),
            key=lambda v: cluster_route_distance(
                distance_matrix, clusters[v] + [outlier_idx]
            )
        )
        clusters[best_vehicle].append(outlier_idx)

    return refine_clusters_by_route_balance(distance_matrix, clusters)


def split_into_balanced_clusters(distance_matrix, num_vehicles=NUM_VEHICLES):
    baseline_clusters = split_into_equal_count_clusters(distance_matrix, num_vehicles=num_vehicles)
    dynamic_clusters = split_into_distance_balanced_clusters(distance_matrix, num_vehicles=num_vehicles)

    base_dist = [cluster_route_distance(distance_matrix, c) for c in baseline_clusters]
    dyn_dist = [cluster_route_distance(distance_matrix, c) for c in dynamic_clusters]
    base_count = [len(c) for c in baseline_clusters]
    dyn_count = [len(c) for c in dynamic_clusters]

    if route_workload_gap(dyn_dist, dyn_count) <= route_workload_gap(base_dist, base_count):
        return dynamic_clusters

    return baseline_clusters

def solve_single_vehicle_route(distance_matrix, node_indices):
    if not node_indices:
        return [0, 0], 0

    kalanlar = set(node_indices)
    route = [0]
    mevcut = 0
    route_distance = 0

    while kalanlar:
        sonraki = min(kalanlar, key=lambda node: int(distance_matrix[mevcut][node]))
        route_distance += int(distance_matrix[mevcut][sonraki])
        route.append(sonraki)
        mevcut = sonraki
        kalanlar.remove(sonraki)

    route_distance += int(distance_matrix[mevcut][0])
    route.append(0)

    return route, route_distance

def cluster_route_distance(distance_matrix, cluster):
    _, route_distance = solve_single_vehicle_route(distance_matrix, cluster)
    return route_distance

def refine_clusters_by_route_balance(distance_matrix, clusters, max_iterations=150):
    clusters = [list(cluster) for cluster in clusters]
    total_nodes = sum(len(cluster) for cluster in clusters)

    for _ in range(max_iterations):
        workloads = []
        for cluster in clusters:
            dist = cluster_route_distance(distance_matrix, cluster)
            workloads.append(get_workload(dist, len(cluster)))

        high_idx = int(np.argmax(workloads))
        low_idx = int(np.argmin(workloads))
        current_gap = workloads[high_idx] - workloads[low_idx]

        if current_gap <= 0:
            break

        best_move = None
        best_gap = current_gap
        high_cluster = clusters[high_idx]
        low_cluster = clusters[low_idx]

        for high_node in list(high_cluster):
            if len(high_cluster) <= 1 and total_nodes >= len(clusters):
                continue

            new_high_cluster = [node for node in high_cluster if node != high_node]
            new_low_cluster = low_cluster + [high_node]

            new_high_workload = get_workload(
                cluster_route_distance(distance_matrix, new_high_cluster),
                len(new_high_cluster)
            )
            new_low_workload = get_workload(
                cluster_route_distance(distance_matrix, new_low_cluster),
                len(new_low_cluster)
            )

            new_gap = abs(new_high_workload - new_low_workload)

            if new_gap < best_gap:
                best_gap = new_gap
                best_move = ('move', high_node)

        if best_move:
            high_node = best_move[1]
            clusters[high_idx].remove(high_node)
            clusters[low_idx].append(high_node)
            continue

        best_swap = None

        for high_node in high_cluster:
            for low_node in low_cluster:
                new_high_cluster = [node if node != high_node else low_node for node in high_cluster]
                new_low_cluster = [node if node != low_node else high_node for node in low_cluster]

                new_high_workload = get_workload(
                    cluster_route_distance(distance_matrix, new_high_cluster),
                    len(new_high_cluster)
                )
                new_low_workload = get_workload(
                    cluster_route_distance(distance_matrix, new_low_cluster),
                    len(new_low_cluster)
                )

                new_gap = abs(new_high_workload - new_low_workload)

                if new_gap < best_gap:
                    best_gap = new_gap
                    best_swap = (high_node, low_node)

        if not best_swap:
            break

        high_node, low_node = best_swap
        high_position = clusters[high_idx].index(high_node)
        low_position = clusters[low_idx].index(low_node)
        clusters[high_idx][high_position] = low_node
        clusters[low_idx][low_position] = high_node

    return clusters

def optimize_balanced_routes(konteynerler, num_vehicles=NUM_VEHICLES):
    distance_matrix = get_osrm_distance_matrix(konteynerler)
    if distance_matrix is None:
        return None, None

    clusters = split_into_balanced_clusters(distance_matrix, num_vehicles=num_vehicles)
    truck_routes = []

    for vehicle_id, cluster in enumerate(clusters, start=1):
        route_indices, route_distance = solve_single_vehicle_route(distance_matrix, cluster)
        route_points = [konteynerler[node_index] for node_index in route_indices]
        truck_routes.append({
            'vehicle_id': vehicle_id,
            'distance': route_distance,
            'pickup_count': len(cluster),
            'route_indices': route_indices,
            'route_points': route_points
        })

    return truck_routes, distance_matrix

def create_route(konteynerler, vardiya_adi):
    print(f"\n{'=' * 60}")
    print(f" {vardiya_adi.upper()} ROTA RAPORU (5 KAMYON)")
    print(f"{'=' * 60}")
    print('OSRM üzerinden karayolu mesafeleri hesaplanıyor...')

    truck_routes, distance_matrix = optimize_balanced_routes(konteynerler, num_vehicles=NUM_VEHICLES)

    if truck_routes is None:
        print('HATA: Karayolu verisi alınamadı. Rota iptal edildi.')
        return []

    toplam_filo_mesafesi = 0
    aktif_mesafeler = []

    for route in truck_routes:
        rota_guzergahi = [konteynerler[index]['id'] for index in route['route_indices']]
        toplam_filo_mesafesi += route['distance']

        print(f"\n🟢 Kamyon {route['vehicle_id']} Detayları:")
        print(f"   Toplanan Konteyner : {route['pickup_count']} adet")
        print(f"   Katedilen Mesafe   : {route['distance']} metre")
        print(f"   Güzergah           : {' -> '.join(rota_guzergahi)}")

        if route['pickup_count'] > 0:
            aktif_mesafeler.append(route['distance'])

    print(f"\n🏁 {vardiya_adi} Toplam Filo Mesafesi: {toplam_filo_mesafesi} metre")

    return truck_routes

if __name__ == '__main__':
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'cop_veri_seti.xlsx')
    model_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')
    scaler_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'doluluk_scaler.pkl')

    filtreli_konteynerler = get_predictions_and_filter(
        veri_yolu,
        model_yolu,
        scaler_yolu=scaler_yolu,
        threshold=40
    )
    sabah_listesi, aksam_listesi = vardiyalara_bol(filtreli_konteynerler)

    sabah_sonuclari = create_route(sabah_listesi, 'Sabah Vardiyası')
    aksam_sonuclari = create_route(aksam_listesi, 'Akşam Vardiyası')

    print("\n Tüm vardiya rotaları hesaplandı.")