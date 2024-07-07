#임시저장용

all_bundles = np.array(all_bundles)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(all_bundles)
    labels = kmeans.labels_

    cluster_0_indices = np.where(labels == 0)[0]
    cluster_1_indices = np.where(labels == 1)[0]

    chosen_cluster_indices = random.choice([cluster_0_indices, cluster_1_indices])

    bundle_indices = random.sample(list(chosen_cluster_indices), 2)

    bundle1 = all_bundles[bundle_indices[0]]
    bundle2 = all_bundles[bundle_indices[1]]
