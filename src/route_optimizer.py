import logging
import math
import os
import pickle
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import folium
import numpy as np
import pandas as pd
import requests

NUM_VEHICLES = 5
WINDOW_SIZE = 5
DISTANCE_TOLERANCE_RATIO = 0.10
OSRM_BASE_URL = 'https://router.project-osrm.org'
ROUTE_COLORS = ['red', 'blue', 'green', 'purple', 'orange']
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
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coords}?annotations=distance"

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


def get_osrm_route_geometry(route_points):
    if len(route_points) < 2:
        return [[route_points[0]['enlem'], route_points[0]['boylam']]] if route_points else []

    coords = ';'.join([f"{point['boylam']},{point['enlem']}" for point in route_points])
    url = f"{OSRM_BASE_URL}/route/v1/driving/{coords}?overview=full&geometries=geojson"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        geometry = data['routes'][0]['geometry']['coordinates']
        return [[lat, lon] for lon, lat in geometry]
    except Exception:
        return [[point['enlem'], point['boylam']] for point in route_points]


def load_prediction_scaler(scaler_yolu, df):
    if scaler_yolu and os.path.exists(scaler_yolu):
        with open(scaler_yolu, 'rb') as scaler_dosyasi:
            return pickle.load(scaler_dosyasi)

    scaler = BasicMinMaxScaler()
    scaler.fit(df[['doluluk_sayisal']])
    return scaler


def get_predictions_and_filter(veri_yolu, model_yolu, scaler_yolu=None, threshold=28):
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


def route_distance_gap(route_distances):
    if not route_distances:
        return 0.0
    ortalama = float(sum(route_distances)) / len(route_distances)
    if ortalama == 0:
        return 0.0
    return max(abs(mesafe - ortalama) / ortalama for mesafe in route_distances)


def split_into_equal_count_clusters(distance_matrix, num_vehicles=NUM_VEHICLES):
    stop_indices = list(range(1, len(distance_matrix)))
    target_sizes = build_target_sizes(len(stop_indices), num_vehicles)
    clusters = [[] for _ in range(num_vehicles)]

    if not stop_indices:
        return clusters

    sorted_stops = sorted(stop_indices, key=lambda idx: distance_matrix[0][idx], reverse=True)

    for vehicle_id in range(min(num_vehicles, len(sorted_stops))):
        if target_sizes[vehicle_id] == 0:
            continue
        seed_node = sorted_stops[vehicle_id]
        clusters[vehicle_id].append(seed_node)

    for node_index in sorted_stops[min(num_vehicles, len(sorted_stops)):]:
        uygun_araclar = [
            vehicle_id for vehicle_id in range(num_vehicles)
            if len(clusters[vehicle_id]) < target_sizes[vehicle_id]
        ]

        best_vehicle = min(
            uygun_araclar,
            key=lambda vehicle_id: (
                cluster_route_distance(distance_matrix, clusters[vehicle_id] + [node_index]),
                len(clusters[vehicle_id])
            )
        )
        clusters[best_vehicle].append(node_index)

    return clusters


def split_into_distance_balanced_clusters(distance_matrix, num_vehicles=NUM_VEHICLES):
    stop_indices = list(range(1, len(distance_matrix)))
    clusters = [[] for _ in range(num_vehicles)]

    if not stop_indices:
        return clusters

    sorted_stops = sorted(stop_indices, key=lambda idx: distance_matrix[0][idx], reverse=True)

    for vehicle_id, seed_node in enumerate(sorted_stops[:min(num_vehicles, len(sorted_stops))]):
        clusters[vehicle_id].append(seed_node)

    for node_index in sorted_stops[min(num_vehicles, len(sorted_stops)):]:
        best_vehicle = min(
            range(num_vehicles),
            key=lambda vehicle_id: cluster_route_distance(distance_matrix, clusters[vehicle_id] + [node_index])
        )
        clusters[best_vehicle].append(node_index)

    return refine_clusters_by_route_balance(distance_matrix, clusters)


def split_into_balanced_clusters(distance_matrix, num_vehicles=NUM_VEHICLES):
    baseline_clusters = split_into_equal_count_clusters(distance_matrix, num_vehicles=num_vehicles)
    dynamic_clusters = split_into_distance_balanced_clusters(distance_matrix, num_vehicles=num_vehicles)

    baseline_distances = [cluster_route_distance(distance_matrix, cluster) for cluster in baseline_clusters]
    dynamic_distances = [cluster_route_distance(distance_matrix, cluster) for cluster in dynamic_clusters]

    if route_distance_gap(dynamic_distances) <= route_distance_gap(baseline_distances):
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


def refine_clusters_by_route_balance(distance_matrix, clusters, max_iterations=80):
    clusters = [list(cluster) for cluster in clusters]
    total_nodes = sum(len(cluster) for cluster in clusters)

    for _ in range(max_iterations):
        route_distances = [cluster_route_distance(distance_matrix, cluster) for cluster in clusters]
        high_idx = int(np.argmax(route_distances))
        low_idx = int(np.argmin(route_distances))
        current_gap = route_distances[high_idx] - route_distances[low_idx]

        if current_gap <= 0:
            break

        best_move = None
        best_gap = current_gap
        high_cluster = clusters[high_idx]
        low_cluster = clusters[low_idx]

        for high_node in list(high_cluster):
            if len(high_cluster) == 1 and total_nodes >= len(clusters):
                continue

            new_high_distance = cluster_route_distance(
                distance_matrix,
                [node for node in high_cluster if node != high_node]
            )
            new_low_distance = cluster_route_distance(distance_matrix, low_cluster + [high_node])
            new_gap = abs(new_high_distance - new_low_distance)

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

                new_high_distance = cluster_route_distance(distance_matrix, new_high_cluster)
                new_low_distance = cluster_route_distance(distance_matrix, new_low_cluster)
                new_gap = abs(new_high_distance - new_low_distance)

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


def save_shift_map(vardiya_adi, truck_routes, cikti_adi=None):
    if not truck_routes:
        return None

    merkez = truck_routes[0]['route_points'][0]
    harita = folium.Map(location=[merkez['enlem'], merkez['boylam']], zoom_start=14, control_scale=True)

    folium.Marker(
        [merkez['enlem'], merkez['boylam']],
        popup='DEPO',
        tooltip='DEPO',
        icon=folium.Icon(color='black', icon='home')
    ).add_to(harita)

    for route in truck_routes:
        color = ROUTE_COLORS[(route['vehicle_id'] - 1) % len(ROUTE_COLORS)]
        geometry = get_osrm_route_geometry(route['route_points'])

        if geometry:
            folium.PolyLine(
                locations=geometry,
                color=color,
                weight=5,
                opacity=0.85,
                tooltip=f"Kamyon {route['vehicle_id']}"
            ).add_to(harita)

        for point in route['route_points'][1:-1]:
            folium.CircleMarker(
                location=[point['enlem'], point['boylam']],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.9,
                popup=(
                    f"<b>Kamyon {route['vehicle_id']}</b><br>"
                    f"Konteyner: {point['id']}<br>"
                    f"Doluluk Tahmini: %{point.get('tahmin_doluluk', 0)}"
                )
            ).add_to(harita)

    if cikti_adi is None:
        cikti_adi = f"optimize_rota_haritasi_{slugify(vardiya_adi)}.html"

    dosya_yolu = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', cikti_adi))
    harita.save(dosya_yolu)
    return dosya_yolu


def save_combined_map(vardiya_sonuclari):
    tum_rotalar = [route for routes in vardiya_sonuclari.values() for route in routes]
    if not tum_rotalar:
        return None

    return save_shift_map('Genel Rota', tum_rotalar, cikti_adi='optimize_rota_haritasi.html')


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

    if aktif_mesafeler:
        ortalama_mesafe = sum(aktif_mesafeler) / len(aktif_mesafeler)
        max_sapma = max(abs(mesafe - ortalama_mesafe) / ortalama_mesafe for mesafe in aktif_mesafeler)
        print(f"\n📊 Mesafe Denge Sapması: %{max_sapma * 100:.1f} (hedef: ≤ %{DISTANCE_TOLERANCE_RATIO * 100:.0f})")

    print(f"\n🏁 {vardiya_adi} Toplam Filo Mesafesi: {toplam_filo_mesafesi} metre")
    harita_yolu = save_shift_map(vardiya_adi, truck_routes)
    if harita_yolu:
        print(f"🗺️ Harita kaydedildi: {harita_yolu}")

    return truck_routes


if __name__ == '__main__':
    veri_yolu = os.path.join(os.path.dirname(__file__), '..', 'data', 'cop_veri_seti.xlsx')
    model_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_doluluk_modeli.keras')
    scaler_yolu = os.path.join(os.path.dirname(__file__), '..', 'models', 'doluluk_scaler.pkl')

    filtreli_konteynerler = get_predictions_and_filter(
        veri_yolu,
        model_yolu,
        scaler_yolu=scaler_yolu,
        threshold=28
    )
    sabah_listesi, aksam_listesi = vardiyalara_bol(filtreli_konteynerler)

    sabah_sonuclari = create_route(sabah_listesi, 'Sabah Vardiyası')
    aksam_sonuclari = create_route(aksam_listesi, 'Akşam Vardiyası')

    genel_harita = save_combined_map({
        'Sabah Vardiyası': sabah_sonuclari,
        'Akşam Vardiyası': aksam_sonuclari
    })
    if genel_harita:
        print(f'🗺️ Genel rota haritası kaydedildi: {genel_harita}')