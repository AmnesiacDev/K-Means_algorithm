import kmeans as km



k = 10
km = km.Kmeans(500, k)
km.update_clusters()
km.compare_seeds(3)
