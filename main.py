#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv # 一度だけ利用
import cv2
import numpy as np
import math
from numpy.linalg import solve
import itertools # http://qiita.com/junkls/items/10384950963056cc8e08
import random
import sys # モジュール属性 argv を取得するため

class MyVector:
    """
    ベクトルや線分の方程式のクラス
    @x1, @y1 座標1
    @x2, @y2 座標2
    """
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        # y = ax + b
        self.a = (float(y2)-float(y1)) / (float(x2)-float(x1))
        self.b = float(y1) - self.a*float(x1)
        # 角度
        self.theta = math.atan(self.a)

    def make_line(self):
        """
        線分の両端をのばして可視化に利用
        """
        mx = 1000 * np.cos(self.theta)
        my = 1000 * np.sin(self.theta)
    
        _x1 = int(self.x1 + mx)
        _y1 = int(self.y1 + my)
        _x2 = int(self.x2 - mx)
        _y2 = int(self.y2 - my)

        return (_x1,_y1,_x2,_y2)

    def to_np_vec(self):
        """
        ベクトルをnumpyのarrayに変換
        """
        vec = np.array((float(self.x2)-float(self.x1), float(self.y2)-float(self.y1)))
        return vec

def mysolve(v1, v2):
    """
    ２元一次連立方程式をとく
    http://d.hatena.ne.jp/sle/20080429/1209466513o
    """
    # y - ax = b
    a = np.array([[1.0, -v1.a], [1.0, -v2.a]])
    b = np.array([v1.b, v2.b])
    return solve(a, b)

def delete_near_points(pts):
    """
    ある座標近くの座標を削除
    """
    result_points = []
    pts_ = pts[:]

    # 座標組み合わせ分ループ
    for p1 in pts:
        # 自分のpointが消去されている時次に行く
        my_pt = [p2 for p2 in pts_ if p1[0] == p2[0] and p1[1] == p1[1]]
        if (len(my_pt) == 0):
            continue

        result_points.append(p1)

        # 自分のpointに近いものを削除
        pts_ = [p2 for p2 in pts_ if np.linalg.norm(p1-p2) > 50]

    return result_points

def draw_points(img, pts):
    """
    複数画像を塗る関数
    """
    color = [int(255*random.random()),int(255*random.random()),int(255*random.random())]

    for x, y in pts:
        draw_pixel(img, x, y, color)
    
def draw_pixel(img, x, y, color):
    """
    画素を塗る関数
    一定の範囲を指定
    """
    a = 6 # 画像を塗る範囲

    for x_ in range(int(x)-a, int(x)+a):
        for y_ in range(int(y)-a, int(y)+a):
            img[x_,y_] = color

def find_rect_from_points(rect_points):
    """
    4つの点の組み合わせが正方形かどうか調べる
    """
    rect = None

    for p1, p2, p3, p4 in list(itertools.permutations(rect_points, 4)):
        if is_square(p1,p2,p3,p4):
            rect = [p1,p2,p3,p4]
            break

    return rect

def is_square(p1,p2,p3,p4):
    """
    4つの点の並びが正方形かどうか調べる
    @ 絹田
    """
    m=(p1+p2+p3+p4)/4

    l1=np.linalg.norm(p1-m)
    l2=np.linalg.norm(p2-m)
    l3=np.linalg.norm(p3-m)
    l4=np.linalg.norm(p4-m)

    lines=np.array([l1,l2,l3,l4])
    ave=np.average(lines)

    sub1=np.linalg.norm(p1-p2)
    sub2=np.linalg.norm(p1-p3)
    sub3=np.linalg.norm(p1-p4)

    subsub=abs(sub1-sub2)
    subsub2=abs(sub1-sub3)
    subsub3=abs(sub2-sub3)

    if (subsub > 20) and (subsub2 > 20):
       return False
    for line in lines:
       #print line
       if (line-ave) > 5 or line < 40:
           return False
   
    return True

def dump_points(pts):
    for x, y in pts:
        print "point"
        print "x" 
        print  x
        print "y:" 
        print  y
    print ""

"""
四角形かどうかのvalidation
座標4点を結んだ線の黒色の割合を調べる
"""
def rect_remap_points(pts):
    """
    四角形の並べ方を右回りか左回りにする
    端の座標を見つけて、ループして見つける
    """
    if (len(pts) != 4):
        print "not rect"
        return pts

    pts_xy_sum = []
    for i in range(4):
        pts_xy_sum.append((i, pts[i][0] + pts[i][1]))

    max_index = max(pts_xy_sum, key=(lambda x: x[1]))[0]
    min_index = min(pts_xy_sum, key=(lambda x: x[1]))[0]
    other_indexes = [i for i in range(4) if i != max_index and i != min_index]

    return [pts[min_index], pts[other_indexes[0]], pts[max_index], pts[other_indexes[1]]]

def rect_whiteness(img, pts):
    # とりあえず一列に並んでおけばok
    # 左上, 右上, 右下, 左下 
    if (len(pts) != 4):
        print "not rect"
        return 0.0

    ave_ratio = 0.0
    for i in range(3):
        ave_ratio += line_whiteness(img, pts[i], pts[i+1])

    return ave_ratio/4

# http://stackoverflow.com/questions/22952792/count-number-of-white-pixels-along-a-line-in-opencv
def line_whiteness(img, pt1, pt2):
    """
    線分の白の割合を調べる
    """

    sum = 0.0
    count = 0.0

    p1 = (int(pt1[1]), int(pt1[0])) # なんか逆
    p2 = (int(pt2[1]), int(pt2[0]))
    li = cv.InitLineIterator(cv.fromarray(img), p1, p2)
    ther_color = 240
    
    for(r,g,b) in li:
        sum += 1
        if(r >= ther_color and g >= ther_color and b >= ther_color):
            count += 1

    return count/sum

if __name__ == '__main__':
    """
    コマンドライン引数あり
    """
    # argvs = sys.argv  # コマンドライン引数を格納したリストの取得

    # if (len(argvs) != 2): 
    #     print 'コマンドライン引数(1)にfilenameを入力'
    #     quit()

    # filename = argvs[1]

    filename = "data/gazoukadai.bmp"


    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    minLineLength = 100
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
    vecs = []
    
    for x1,y1,x2,y2 in lines[0]:
        vec = MyVector(x1,y1,x2,y2)
        vecs.append(vec)
        # 認識した線分の描画
        # _x1, _y1, _x2, _y2 = vec.make_line()
        # cv2.line(img,(_x1,_y1),(_x2,_y2),(0,255,0),2)
    
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
    pts = delete_near_points(pts)
    print('直交座標の数: {0}'.format(len(pts)))
    
    # 全ての並びに対して調べる
    # 組み合わせの順列 = 順列の上から4つ 
    # NOTE!!: 計算に1分ぐらい時間がかかる!!
    rects = []

    for rect_points in list(itertools.combinations(pts, 4)):
        rect = find_rect_from_points(rect_points)

        if rect is not None: 
            rects.append(rect)
        
    print('正方形の数: {0}'.format(len(rects)))

    results = [] # テスト用

    for rect_pts in rects:
        # 座標の並び方をremap
        rect_pts = rect_remap_points(rect_pts)

        if (rect_whiteness(img, rect_pts) < 0.2): # 白色の割合が少ない
            results.append(rect_pts)
            draw_points(img, rect_pts)

    print('正方形の数: {0}'.format(len(results)))

    cv2.imwrite('./gazoukadai.jpg',img)
    
