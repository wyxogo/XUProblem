import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def build_tsne(feature_data, label, title,arg):
    '''Create T-SNE figure
    
    Args:
        feature_data (list): ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
                            If the metric is 'precomputed' X must be a square distance
                            matrix. Otherwise it contains a sample per row. If the method
                            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
                            or 'coo'. If the method is 'barnes_hut' and the metric is
                            'precomputed', X may be a precomputed sparse graph.
        label (list): Data labels
        title (string): Figure title
        
    '''
    t_sne = TSNE(n_components=2, init='pca', random_state=0)
    data = t_sne.fit_transform(feature_data)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    
    plt.savefig(f'{arg.output}{title}.jpg',dpi=300)
    plt.show()
