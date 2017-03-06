import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
import imp
from mpl_toolkits.mplot3d import Axes3D
io = imp.load_source('io', 'code/common/io.py')

ls = []
for line in open('forward.rels'):
    dp = line.strip().split('\t')
    dp = [float(d) for d in dp]
    ls.append(dp)

X = np.array(ls)

#n_clusters=2
#k_means = KMeans(init='k-means++', n_clusters=n_clusters)
#k_means.fit(X)

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
#colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'red', 'green', 'blue']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
#k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
#k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

# KMeans
ax = fig.gca(projection='3d')
'''
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
'''
#ax.set_title('Degrees')
#ax.set_xticks(())
#ax.set_yticks(())
ax.set_aspect('equal')
ax.grid(True, which='both')
ax.axis([-2, 2, -2, 2])

names = []
for line in open('data/FB15k/relations.dict'):
    names.append(line.strip().split('\t')[1])

source_triplets = list(io.read_triplets('data/FB15k/train.txt'))

d = {k:0 for k in names}
for t in source_triplets:
    d[t[1]] += 1


plt.get_cmap('plasma')

t = [d[name] for name in names]

print(names[np.argmax(t)])
print(np.max(t))
print(X[np.argmax(t)])


ax.scatter(X[:,0], X[:,1], t, marker='o', alpha=1, s=5)


ax.set_xlabel('Basis 1')
ax.set_ylabel('Basis 2')
ax.set_zlabel('# Edges')
#ax.set_zscale('log')

#ax.set_zticks([20, 200, 500])

#ax.colorbar()

# loop through each x,y pair
'''for coords, name in zip(X, names):
    if d[name] < 50:
        colstring = 'yellow'
    elif d[name] < 1000:
        colstring = 'orange'
    else:
        colstring = 'red'

    ax.annotate(str(d[name]),  xy=(coords[0], coords[1]), color=colstring)
'''
plt.show()
