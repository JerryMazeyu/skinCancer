import cv2
import numpy as np

class ProcessImage(object):
    """
    处理图像
    """

    def _0ShowImg(self, img):
        """展示图像"""
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _1BGR2GRAY(self, img):
        """BGR转灰色"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def _2GRAY2BGR(self, img):
        """灰色转GBR"""
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _3Closing(self, img):
        """去掉毛发的腐蚀操作"""
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.dilate(closing, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)
        return img

    def _4OTSU(self, img):
        """自适应二分"""
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2 = 255-th2
        return th2

    def _5FindMaxContourAndCalSimilarity(self, img, verbose=False):
        """
        找到最大的边界并通过拟合椭圆->计算相似度来评价其对称性
        """
        cnt, _ = cv2.findContours(img, 3, 2)
        s_list = []
        for ind, cont in enumerate(cnt):
            area = cv2.contourArea(cont)
            tmp = (ind, area)
            s_list.append(tmp)
        s_list.sort(key=lambda x: x[1], reverse=True)
        tar_cnt = cnt[s_list[0][0]]
        img_bgr = self._2GRAY2BGR(img)
        img_blk = np.zeros_like(img_bgr)
        ellipse = cv2.fitEllipse(tar_cnt)
        cv2.ellipse(img_bgr, ellipse, (255, 255, 0), 2)
        cv2.ellipse(img_blk, ellipse, (255, 255, 0), 2)
        img_blk = self._1BGR2GRAY(img_blk)
        el_cnt, _ = cv2.findContours(img_blk, 3, 2)
        if verbose:
            cv2.drawContours(img_bgr, [tar_cnt], 0, (0, 0, 255), 2)
            self._0ShowImg(img_bgr)
            print("-->", cv2.matchShapes(tar_cnt, el_cnt[0], 1, 0.0)-1e308)
        return np.clip(cv2.matchShapes(tar_cnt, el_cnt[0], 1, 0.0), 0, 1)

    def _8SaveImgs(self, img, name='saved1.jpg'):
        """存储图像"""
        cv2.imwrite(name, img)

    def __call__(self, imgpath):
        img = cv2.imread(imgpath)
        try:
            img = self._1BGR2GRAY(img)
            img = self._3Closing(img)
            img = self._4OTSU(img)
            res = self._5FindMaxContourAndCalSimilarity(img)
        except:
            res = 0
        return res





if __name__ == '__main__':
    prim = ProcessImage()
    imgpath = '/Users/mazeyu/newEraFrom2020.5/skinCancer/data/train-2c/melanoma/ISIC_0000296.jpg'
    prim(imgpath)
