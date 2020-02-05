import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
def generate_tsne(path=None, size=(100, 100), word_count=1000,words=None, embeddings=None):
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=50,n_components=2, init='pca', n_iter=100000,random_state=32,verbose=1)
    low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
    labels = words[:word_count]
    return _plot_with_labels(low_dim_embs, labels, path, size)
def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    X = low_dim_embs
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = labels
    figure = plt.figure(figsize=size) # in inches
    df['tsne-2d-one'] = low_dim_embs[:,0]
    df['tsne-2d-two'] = low_dim_embs[:,1]
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=1
    )

    if path is not None:
        figure.savefig(path)
        plt.close(figure)

words = []
embeddings = []
pred_lb = []
with open('test_pred.txt','r',encoding="utf-8") as lines:
    for line in lines:
        pred_lb.append(line.strip().split('\t')[1])
i = 0
with open('cls.txt','r',encoding='utf-8') as lines:
    for line in lines:
        tmp = line.strip().split('\t')
        words.append(tmp[0])
        t_v = []
        for i in range(1,len(tmp)):
            t_v.append(float(tmp[1]))
        embeddings.append(t_v)
embeddings= np.array(embeddings)
generate_tsne(path='./log1', size=(16,9), word_count=3000,words=pred_lb, embeddings=embeddings)
