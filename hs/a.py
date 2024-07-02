from sklearn.cluster import KMeans
import time
import numpy as np
from util import *

def initial_bundles_with_kmeans(all_orders, num_clusters):
    # 주문들의 위치를 기반으로 군집화
    order_locations = np.array([[order.location_x, order.location_y] for order in all_orders])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(order_locations)

    clusters = kmeans.labels_
    bundles = [[] for _ in range(num_clusters)]

    for i, order in enumerate(all_orders):
        bundles[clusters[i]].append(order)

    return bundles


def try_merging_bundles(K, dist_mat, bundle1, bundle2):
    # 병합 가능성 체크: 두 번들의 용량이 라이더의 제한을 초과하지 않는지 확인
    total_volume = bundle1.volume + bundle2.volume
    if total_volume > max(bundle1.rider.capacity, bundle2.rider.capacity):
        return None

    # 병합 후의 경로 및 비용 계산 (여기서는 단순한 합으로 가정)
    new_bundle_orders = bundle1.orders + bundle2.orders

    # 새로운 번들의 비용 계산
    new_cost = calculate_bundle_cost(K, dist_mat, new_bundle_orders)
    old_cost = bundle1.cost + bundle2.cost

    if new_cost < old_cost:
        new_bundle = Bundle(
            orders=new_bundle_orders,
            rider=bundle1.rider,  # 임의로 bundle1의 라이더를 사용, 나중에 변경 가능
            shop_seq=[order.shop_id for order in new_bundle_orders],
            dlv_seq=[order.id for order in new_bundle_orders],
            volume=total_volume,
            cost=new_cost
        )
        return new_bundle
    else:
        return None


def calculate_bundle_cost(K, dist_mat, bundle_orders):
    # 새로운 번들의 비용 계산 로직
    # 거리와 기타 요인들을 고려한 비용 계산
    shop_seq = [order.shop_id for order in bundle_orders]
    dlv_seq = [order.id for order in bundle_orders]

    cost = 0
    for i in range(len(shop_seq) - 1):
        cost += dist_mat[shop_seq[i], shop_seq[i + 1]]
    for j in range(len(dlv_seq) - 1):
        cost += dist_mat[dlv_seq[j], dlv_seq[j + 1]]
    return cost


def select_two_bundles(all_bundles):
    # 두 개의 번들을 선택하는 로직 (여기서는 단순 랜덤 선택)
    import random
    bundle1, bundle2 = random.sample(all_bundles, 2)
    return bundle1, bundle2


def algorithm(K, all_orders, all_riders, dist_mat, timelimit=60):
    start_time = time.time()

    for r in all_riders:
        r.T = np.round(dist_mat / r.speed + r.service_time)

    # A solution is a list of bundles
    solution = []

    # ------------- Custom algorithm code starts from here --------------#

    num_clusters = len(all_riders)  # 라이더 수만큼 군집 생성
    initial_bundles = initial_bundles_with_kmeans(all_orders, num_clusters)

    all_bundles = []
    for i, bundle_orders in enumerate(initial_bundles):
        new_bundle = Bundle(
            orders=bundle_orders,
            rider=all_riders[i],
            shop_seq=[order.shop_id for order in bundle_orders],
            dlv_seq=[order.id for order in bundle_orders],
            volume=sum(order.volume for order in bundle_orders),
            cost=calculate_bundle_cost(K, dist_mat, bundle_orders)
        )
        all_bundles.append(new_bundle)
        all_riders[i].available_number -= 1

    best_obj = sum((bundle.cost for bundle in all_bundles)) / K
    print(f'Best obj = {best_obj}')

    # 번들 병합 알고리즘
    while True:

        iter = 0
        max_merge_iter = 1000

        while iter < max_merge_iter:

            bundle1, bundle2 = select_two_bundles(all_bundles)
            new_bundle = try_merging_bundles(K, dist_mat, bundle1, bundle2)

            if new_bundle is not None:
                all_bundles.remove(bundle1)
                bundle1.rider.available_number += 1

                all_bundles.remove(bundle2)
                bundle2.rider.available_number += 1

                all_bundles.append(new_bundle)
                new_bundle.rider.available_number -= 1

                cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    print(f'Best obj = {best_obj}')

            else:
                iter += 1

            if time.time() - start_time > timelimit:
                break

        if time.time() - start_time > timelimit:
            break

        for bundle in all_bundles:
            new_rider = get_cheaper_available_riders(all_riders, bundle.rider)
            if new_rider is not None:
                old_rider = bundle.rider
                if try_bundle_rider_changing(dist_mat, bundle, new_rider):
                    old_rider.available_number += 1
                    new_rider.available_number -= 1

                if time.time() - start_time > timelimit:
                    break

        cur_obj = sum((bundle.cost for bundle in all_bundles)) / K
        if cur_obj < best_obj:
            best_obj = cur_obj
            print(f'Best obj = {best_obj}')

    # Solution is a list of bundle information
    solution = [
        # rider type, shop_seq, dlv_seq
        [bundle.rider.type, bundle.shop_seq, bundle.dlv_seq]
        for bundle in all_bundles
    ]

    # ------------- End of custom algorithm code--------------#

    return solution
