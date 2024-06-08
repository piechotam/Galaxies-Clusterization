import random

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def count_wcss_scores(X, k_max):
    scores = []
    for k in range(1, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        wcss = kmeans.score(X) * -1 # score returns -WCSS
        scores.append(wcss)
    return scores

def count_clustering_scores(X, cluster_num, model, score_fun):
    if isinstance(cluster_num, int):
        cluster_num_iter = [cluster_num]
    else:
        cluster_num_iter = cluster_num
        
    scores = []    
    for k in cluster_num_iter:
        model_instance = model(n_clusters=k)
        labels = model_instance.fit_predict(X)
        wcss = score_fun(X, labels)
        scores.append(wcss)
    
    if isinstance(cluster_num, int):
        return scores[0]
    else:
        return scores

def create_groups(filenames, data, labels):
    groups = {} # cluster_id : images
    for file, cluster in zip(filenames[:len(data)], labels):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    return groups

def plot_clusters(groups):
    n_clusters = len(groups.keys())

    fig, axs = plt.subplots(n_clusters, 5, figsize=(15, 15))
    for i in range(n_clusters):
        for j in range(5):
            ax = axs[i, j]
            ax.axis('off')
            filename = f'../data/images/{groups[i][random.randint(0, len(groups[i]) - 1)]}'
            ax.imshow(plt.imread(filename))
            ax.set_title(f"Cluster {i}")

    plt.show()

def plot_random_images_from_cluster(cluster, groups):
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    cluster_images = groups[cluster]
    random_images = random.sample(cluster_images, 16)
    
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.axis('off')
            filename = f'../data/images/{random_images[i*4+j]}'
            ax.imshow(plt.imread(filename))
            ax.set_title(f"Cluster {cluster}")
    
    plt.show()