#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from numpy.linalg import solve
# http://qiita.com/junkls/items/10384950963056cc8e08
import itertools

class MyVector:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.a = (float(y2)-float(y1)) / (float(x2)-float(x1))
        self.b = float(y1) - self.a*float(x1)
        self.theta = math.atan(self.a)

    def make_line(self):
        mx = 1000 * np.cos(self.theta)
        my = 1000 * np.sin(self.theta)
    
        _x1 = int(self.x1 + mx)
        _y1 = int(self.y1 + my)
        _x2 = int(self.x2 - mx)
        _y2 = int(self.y2 - my)

        return (_x1,_y1,_x2,_y2)

    def to_np_vec(self):
        vec = np.array((float(self.x2)-float(self.x1), float(self.y2)-float(self.y1)))
        return vec

# ２元一次連立方程式をとく
# http://d.hatena.ne.jp/sle/20080429/1209466513
def mysolve(v1, v2):
    # y - ax = b
    a = np.array([[1.0, -v1.a], [1.0, -v2.a]])
    b = np.array([v1.b, v2.b])
    return solve(a, b)

def delete_near_points(pts):
    # 組み合わせ分ループ
    for p1, p2 in list(itertools.combinations(list(pts),2))

def draw_pixel(img, x, y):
    a = 5 # 画像を塗る範囲

    color = [0,0,255]
    for x_ in range(int(x)-a, int(x)+a):
        for y_ in range(int(y)-a, int(y)+a):
            img[x_,y_] = color



img = cv2.imread('./gazoukadai5.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

minLineLength = 100
maxLineGap = 5
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
print lines

vecs = []

for x1,y1,x2,y2 in lines[0]:
    vec = MyVector(x1,y1,x2,y2)
    # cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    _x1, _y1, _x2, _y2 = vec.make_line()
    cv2.line(img,(_x1,_y1),(_x2,_y2),(0,255,0),2)
    vecs.append(vec)

# 全ての線分の内積を計算
dots = [(v1, v2, np.dot(v1.to_np_vec(), v2.to_np_vec())) for v1 in vecs for v2 in vecs]

# 許容範囲を元に直交した座標を取得
pts = []
threshold = 0.5

for dot in dots:
    dot_ = dot[2]
    if (-threshold <= dot_ and dot_ <= threshold):
        v1 = dot[0]
        v2 = dot[1]
        pts.append(mysolve(v1, v2))


# 直交座標の量を減らす
print len(pts)

color = [0,0,255]
for x, y in pts:
    draw_pixel(img, x, y)

cv2.imwrite('./gazoukadai.jpg',img)

