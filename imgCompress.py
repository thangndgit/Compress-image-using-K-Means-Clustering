# image processing
from PIL import Image
from io import BytesIO
from PyQt5.QtGui import QImage, QPixmap

# data analysis
import math
import numpy as np
import pandas as pd

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
        self.k_means_res = None
        self.ori_img_size = None
        self.res_img_size = None
        self.ori_img_n_colors = None

    def setOriImg(self, oriImg):
        self.ori_img = oriImg
        self.X = np.array(oriImg.getdata())
        self.ori_img_size = self.imgByteSize(oriImg)
        self.ori_img_n_colors = len(set(oriImg.getdata()))
        self.res_img = None
        self.res_pixmap = None
        self.k_means_res = None
    
    def setKMeans(self, nbClus, maxIter = 10):
        self.nb_clus = nbClus
        self.k_means = KMeans(n_clusters = nbClus, max_iter=maxIter).fit(self.X)
    
    def runKMeans(self):
        new_pixels = self.replaceWithCentroid(self.k_means)
        WCSS = self.k_means.inertia_
        BCSS = self.calculateBCSS(self.X, self.k_means)
        self.exp_var = 100*BCSS/(WCSS+BCSS)
        self.res_img = Image.fromarray(np.uint8(new_pixels))
        self.res_img_size = self.imgByteSize(self.res_img)
        new_pixels = new_pixels[:,:,:3]
        new_pixels = new_pixels.astype(np.uint8)
        height, width, _ = new_pixels.shape
        bytesPerLine = 3 * width
        qImg = QImage(new_pixels.data, width, height, bytesPerLine, QImage.Format_RGB888)
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
    
    def calculateKMeansResult(self):
        kmax = 21
        if(self.ori_img_n_colors < 21):
            kmax = self.ori_img_n_colors
        range_k_cluster = (2, kmax)
        kmeans_result = []
        for k in range(*range_k_cluster):
            # CLUSTERING
            k_means_k = KMeans(n_clusters = k).fit(self.X)

            # REPLACE PIXELS WITH ITS CENTROID
            new_pixels = self.replaceWithCentroid(k_means_k)

            # EVALUATE
            WCSS = k_means_k.inertia_
            BCSS = self.calculateBCSS(self.X, k_means_k)
            exp_var = 100*BCSS/(WCSS+BCSS)
            metric = {
                "K": k,
                "EV": exp_var,
                "IS": self.imgByteSize(new_pixels)
            }
            kmeans_result.append(metric)

        kmeans_result = pd.DataFrame(kmeans_result).set_index("K")
        self.k_means_res = kmeans_result
    
    def calculateBCSS(self, X, kmeans):
        _, label_counts = np.unique(kmeans.labels_, return_counts = True)
        diff_cluster_sq = np.linalg.norm(kmeans.cluster_centers_ - np.mean(X, axis = 0), axis = 1)**2
        return sum(label_counts * diff_cluster_sq)

    def locateOptimalElbow(self, x, y):
        # START AND FINAL POINTS
        p1 = (x[0], y[0])
        p2 = (x[-1], y[-1])
        
        # EQUATION OF LINE: y = mx + c
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = (p2[1] - (m * p2[0]))
        
        # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
        a, b = m, -1
        dist = np.array([abs(a*x0+b*y0+c)/math.sqrt(a**2+b**2) for x0, y0 in zip(x,y)])
        return np.argmax(dist) + x[0]