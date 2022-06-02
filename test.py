import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np
import warnings;
from sklearn.cluster import MiniBatchKMeans
from PIL import Image



def tinhToan_ChePhu(url_image):
    sns.set()  # for plot styling
    img = Image.open(url_image)

    ax = plt.axes(xticks=[], yticks=[])

    ax.imshow(img);

    img_rgb = img.convert("RGB")

    img_arr = np.array(img_rgb)

    img_data = img_arr / 255.0  # use 0...1 scale
    img_data = img_data.reshape(img_arr.shape[0] * img_arr.shape[1], img_arr.shape[2])

    n_subnet = (np.floor(1037919 / 10 ** 5) * 10 ** 5) / 100
    n_subnet = int(n_subnet)

    def plot_pixels(data, title, colors=None, N=10000):
        if colors is None:
            colors = data

        rng = np.random.RandomState(0)
        i = rng.permutation(data.shape[0])[
            :N]  # https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html, https://www.w3schools.com/python/numpy/numpy_random_permutation.asp
        colors = colors[i]
        R, G, B = data[i].T

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        ax[0].scatter(R, G, color=colors, marker='.')
        ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

        ax[1].scatter(R, B, color=colors, marker='.')
        ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
        # Thêm một tiêu đề trung tâm vào hình.

        fig.suptitle(title, size=20);

    plot_pixels(
        data=img_data,
        title=f'Biểu đồ thể hiện không gian màu: hình gốc (từ tập dữ liệu img)',
        N=int(n_subnet)
    )

    warnings.simplefilter('ignore')  # Fix NumPy issues.

    clts = MiniBatchKMeans(3)

    clts.fit(img_data)

    lables = clts.predict(img_data)

    img_clts = clts.cluster_centers_[lables]

    plot_pixels(
        data=img_data,
        colors=img_clts,
        title="Biểu đồ thể hiện không gian màu: phân cụm"
    )
    y_kmeans = clts.predict(img_data)

    centers = clts.cluster_centers_
    plt.scatter(img_data[:, 0], img_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='orange', s=200, alpha=0.5);
    plt.show()
    clts.labels_
    img_recolored = img_clts.reshape(img_arr.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)

    ax[0].imshow(img)
    ax[0].set_title('Ảnh ban đầu', size=16)
    ax[1].imshow(img_recolored)
    ax[1].set_title('Ảnh kết quả', size=16);

    labels = clts.labels_
    print(labels)
    labels = list(labels)

    centroid = clts.cluster_centers_
    print(centroid)

    def TyLe(labels, centroid):
        percent = []
        for i in range(len(centroid)):
            j = labels.count(i)
            j = j / (len(labels))
            percent.append(j)
        print(percent)
        return percent

    tyle_phantram = TyLe(labels, centroid)
    tl_che_phu = round((tyle_phantram[0] + tyle_phantram[2]) * 100, 3)
    plot_pixels(
        data=img_data,
        colors=img_clts,
        title="Biểu đồ thể hiện không gian màu: phân cụm"
    )
    return tl_che_phu

