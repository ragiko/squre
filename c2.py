# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt

## ------------------------
## BGRからRGBに変換
## ------------------------
def bgr2rbg(im):
    b,g,r = cv2.split(im)
    im = cv2.merge([r,g,b])
    return im
 
## ------------------------
## 結果表示
## ------------------------
def show_result(im1,im2,im3,im4):
 
    graph = plt.figure()
    plt.rcParams["font.size"]=15
    # 入力画像
    plt.subplot(2,2,1),plt.imshow(bgr2rbg(im1))
    plt.title("Input Image")
    # 出力画像
    plt.subplot(2,2,2),plt.imshow(bgr2rbg(im2))
    plt.title("Output Image")
    # 2値化画像
    plt.subplot(2,2,3),plt.imshow(im3,"gray")
    plt.title("Threshold Image")
    # 輪郭画像
    plt.subplot(2,2,4),plt.imshow(im4,"gray")
    plt.title("Edge Image")
    plt.show()
 
 
## ------------------------
## メイン
## ------------------------
if __name__ == '__main__':
 
    fn="gazoukadai_part.bmp"
    # 入力画像の取得
    im_in = cv2.imread(fn)
    for i in range(100):
        im_in[i+10,i+10] = [254,0,0]    
    im_out = cv2.imread(fn)
    # 入力画像をグレースケール変換
    im_gray = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)
 
    # グレースケール画像を2値化
    im_th = cv2.adaptiveThreshold(im_gray,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # 2値化画像からエッジを検出
    im_edge = cv2.Canny(im_th,50,150,apertureSize = 3)
    # エッジ画像から直線の検出
    lines = cv2.HoughLines(im_edge,2,np.pi/180,200)

 
    # 直線の描画
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(im_out,(x1,y1),(x2,y2),(0,0,255),2)
 
    # 画像表示
    show_result(im_in,im_out,im_th,im_edge)
