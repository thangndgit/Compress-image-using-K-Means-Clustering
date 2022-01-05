# image processing
from PIL import Image
from io import BytesIO
from PyQt5.QtGui import QImage, QPixmap

# data analysis
import numpy as np
import matplotlib.pyplot as plt

# modeling
from sklearn.cluster import KMeans

class imgSeg:
    def __init__(self):
        self.X = None
        self.exp_var = None
        self.k_means = None
        self.nb_clus = None
        self.ori_img = None
        self.res_img = None
        self.res_pixmap = None
        self.ori_img_size = None
        self.res_img_size = None
        self.ori_img_n_colors = None

    def setOriImg(self, oriImg):
        self.ori_img = Image.new('RGB', oriImg.size, (255, 255, 255))
        self.ori_img.paste(oriImg, None)
        self.X = np.array(self.ori_img.getdata())
        self.ori_img_size = self.imgByteSize(self.ori_img)
        self.ori_img_n_colors = len(set(self.ori_img.getdata()))
        self.res_img = None
        self.res_pixmap = None
        self.k_means_res = None
    
    def setKMeans(self, nbClus):
        self.nb_clus = nbClus
        self.k_means = KMeans(n_clusters = nbClus, max_iter = 64).fit(self.X)
    
    def drawOptimalElbowPlot(self, opfrom, opto):
        wcss = []
        bcss = []
        evar = []

        K = range(opfrom, opto)

        for k in K:
            kmeans = KMeans(n_clusters = k, max_iter = 64).fit(self.X)
            wcssi = kmeans.inertia_
            bcssi = self.calculateBCSS(self.X, kmeans)
            evari = 100*bcssi/(bcssi+wcssi)
            wcss.append(wcssi)
            bcss.append(bcssi)
            evar.append(evari)

        plt.figure(figsize=(8, 4))
        plt.plot(K, wcss, 'bx-')
        plt.xlabel('K')
        plt.ylabel('WCSS')
        plt.title('The elbow method showing the optimal K of WCSS')
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(K, bcss, 'bx-')
        plt.xlabel('K')
        plt.ylabel('BCSS')
        plt.title('The elbow method showing the optimal K of BCSS')
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(K, evar, 'bx-')
        plt.xlabel('K')
        plt.ylabel('Explained variance')
        plt.title('The elbow method showing the optimal K of Explained variance')
        plt.show()
    
    def runKMeans(self):
        new_pixels = self.replaceWithCentroid(self.k_means)
        WCSS = self.k_means.inertia_
        BCSS = self.calculateBCSS(self.X, self.k_means)
        self.exp_var = 100*BCSS/(WCSS+BCSS)
        self.res_img = Image.fromarray(np.uint8(new_pixels))
        self.res_img_size = self.imgByteSize(self.res_img)
        new_pixels = new_pixels.astype(np.uint8)
        height, width, _ = new_pixels.shape
        qImg = QImage(new_pixels.data, width, height, 3*width, QImage.Format_RGB888)
        self.res_pixmap = QPixmap(qImg)
    
    def imgByteSize(self, img):
        img_file = BytesIO()
        image = Image.fromarray(np.uint8(img))
        image.save(img_file, 'png')
        return img_file.tell()/1024
    
    def replaceWithCentroid(self, kmeans):
        new_pixels = []
        width, height = self.ori_img.size
        for label in kmeans.labels_:
            pixel_as_centroid = list(kmeans.cluster_centers_[label])
            new_pixels.append(pixel_as_centroid)
        new_pixels = np.array(new_pixels).reshape(height, width, -1)
        return new_pixels
        
    def calculateBCSS(self, X, kmeans):
        _, label_counts = np.unique(kmeans.labels_, return_counts = True)
        diff_cluster_sq = np.linalg.norm(kmeans.cluster_centers_ - np.mean(X, axis = 0), axis = 1)**2
        return sum(label_counts * diff_cluster_sq)