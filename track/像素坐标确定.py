import numpy as np
import pandas as pd
import math
import cv2
import datetime

"""
1、按照东南西北的顺序
2、先标四个进口道，从靠近交叉口处按每个车道Z字标点
3、出口道标四个点（可以标大一点）
4、记一下东南西北四个进口的车道数
"""

tpPointsChoose = []
pointsCount = 0


world = []
video = []


def on_mouse(event, x, y, flags, param):
    global frame, point1, world
    global tpPointsChoose  # 存入选择的点
    global pointsCount  # 对鼠标按下的点计数
    global img2, ROI_bymouse_flag
    img2 = frame.copy()  # 此行代码保证每次都重新再原图画  避免画多了
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        pointsCount = pointsCount + 1
        point1 = (x, y)
        print(point1, ',')
        # 画出点击的点
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)
        # world_point = input('请输入' + str(point1) + '对应的世界坐标' + 'x,y')
        # world_point = world_point.split(',')
        # world_point = [float(i) for i in world_point]
        # world.append(world_point)
        # 将选取的点保存到list列表里
        tpPointsChoose.append((x, y))  # 用于画点
        # 将鼠标选的点用直线连起来
        for i in range(len(tpPointsChoose) - 1):
            # print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
        # ----------点击到pointMax时可以提取去绘图----------------
        cv2.namedWindow('src', cv2.WINDOW_NORMAL)
        cv2.imshow('src', img2)
    # -------------------------双击 结束选取-----------------------------
    if event == cv2.EVENT_RBUTTONDOWN:
        # if event == cv2.EVENT_LBUTTONDBLCLK:
        # -----------绘制感兴趣区域-----------
        ROI_byMouse()
        ROI_bymouse_flag = 1
        lsPointsChoose = []


def ROI_byMouse():
    global world, video
    world = np.array(world)
    video = tpPointsChoose
    video = np.array(video)
    # print(world, video)


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1,
                       0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0,
                           A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(
        warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMat

def Trans(img_coordinate, M):
    w_coordinate = np.dot(M, np.array(
        [img_coordinate[0], img_coordinate[1], 1]).T)
    world_coordinate = [w_coordinate[0] / w_coordinate[2],
                        w_coordinate[1] / w_coordinate[2]]
    return world_coordinate
# camera = cv2.VideoCapture('./data/output.mp4')
camera = cv2.VideoCapture(r'F:\66-82\N66-1.MP4')
num = 0
while True:
    res, frame = camera.read()
    if num < 1:
        num += 1
        continue
    if not res:
        break
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', frame)
    if cv2.waitKey(1) == ord('q'):

        break
camera.release()
cv2.destroyAllWindows()  # 关闭所有窗口
cv2.waitKey(1)

kkkk = frame.copy()
cv2.namedWindow('src', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('src', on_mouse)

cv2.imshow('src', kkkk)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
print(f" enter_east ='{int(tpPointsChoose[0][0]*1278/1915)},{int(tpPointsChoose[0][1]*717/1078)},{int(tpPointsChoose[1][0]*1278/1915)},{int(tpPointsChoose[1][1]*717/1078)},{int(tpPointsChoose[2][0]*1278/1915)},{int(tpPointsChoose[2][1]*717/1078)},{int(tpPointsChoose[3][0]*1278/1915)},{int(tpPointsChoose[3][1]*717/1078)}'\n"
      f" enter_west ='{int(tpPointsChoose[4][0]*1278/1915)},{int(tpPointsChoose[4][1]*717/1078)},{int(tpPointsChoose[5][0]*1278/1915)},{int(tpPointsChoose[5][1]*717/1078)},{int(tpPointsChoose[6][0]*1278/1915)},{int(tpPointsChoose[6][1]*717/1078)},{int(tpPointsChoose[7][0]*1278/1915)},{int(tpPointsChoose[7][1]*717/1078)}'\n"
      f" enter_south ='{int(tpPointsChoose[8][0]*1278/1915)},{int(tpPointsChoose[8][1]*717/1078)},{int(tpPointsChoose[9][0]*1278/1915)},{int(tpPointsChoose[9][1]*717/1078)},{int(tpPointsChoose[10][0]*1278/1915)},{int(tpPointsChoose[10][1]*717/1078)},{int(tpPointsChoose[11][0]*1278/1915)},{int(tpPointsChoose[11][1]*717/1078)}'\n"
      f" enter_north ='{int(tpPointsChoose[12][0]*1278/1915)},{int(tpPointsChoose[12][1]*717/1078)},{int(tpPointsChoose[13][0]*1278/1915)},{int(tpPointsChoose[13][1]*717/1078)},{int(tpPointsChoose[14][0]*1278/1915)},{int(tpPointsChoose[14][1]*717/1078)},{int(tpPointsChoose[15][0]*1278/1915)},{int(tpPointsChoose[15][1]*717/1078)}'\n"
      f" exit_east ='{int(tpPointsChoose[16][0]*1278/1915)},{int(tpPointsChoose[16][1]*717/1078)},{int(tpPointsChoose[17][0]*1278/1915)},{int(tpPointsChoose[17][1]*717/1078)},{int(tpPointsChoose[18][0]*1278/1915)},{int(tpPointsChoose[18][1]*717/1078)},{int(tpPointsChoose[19][0]*1278/1915)},{int(tpPointsChoose[19][1]*717/1078)}'\n"
      f" exit_west ='{int(tpPointsChoose[20][0]*1278/1915)},{int(tpPointsChoose[20][1]*717/1078)},{int(tpPointsChoose[21][0]*1278/1915)},{int(tpPointsChoose[21][1]*717/1078)},{int(tpPointsChoose[22][0]*1278/1915)},{int(tpPointsChoose[22][1]*717/1078)},{int(tpPointsChoose[23][0]*1278/1915)},{int(tpPointsChoose[23][1]*717/1078)}'\n"
      f" exit_south ='{int(tpPointsChoose[24][0]*1278/1915)},{int(tpPointsChoose[24][1]*717/1078)},{int(tpPointsChoose[25][0]*1278/1915)},{int(tpPointsChoose[25][1]*717/1078)},{int(tpPointsChoose[26][0]*1278/1915)},{int(tpPointsChoose[26][1]*717/1078)},{int(tpPointsChoose[27][0]*1278/1915)},{int(tpPointsChoose[27][1]*717/1078)}'\n"
      f" exit_north ='{int(tpPointsChoose[28][0]*1278/1915)},{int(tpPointsChoose[28][1]*717/1078)},{int(tpPointsChoose[29][0]*1278/1915)},{int(tpPointsChoose[29][1]*717/1078)},{int(tpPointsChoose[30][0]*1278/1915)},{int(tpPointsChoose[30][1]*717/1078)},{int(tpPointsChoose[31][0]*1278/1915)},{int(tpPointsChoose[31][1]*717/1078)}'\n"
      )
print(f" enter_east ='{int(tpPointsChoose[0][0])},{int(tpPointsChoose[0][1])},{int(tpPointsChoose[1][0])},{int(tpPointsChoose[1][1])},{int(tpPointsChoose[2][0])},{int(tpPointsChoose[2][1])},{int(tpPointsChoose[3][0])},{int(tpPointsChoose[3][1])}'\n"
      f" enter_west ='{int(tpPointsChoose[4][0])},{int(tpPointsChoose[4][1])},{int(tpPointsChoose[5][0])},{int(tpPointsChoose[5][1])},{int(tpPointsChoose[6][0])},{int(tpPointsChoose[6][1])},{int(tpPointsChoose[7][0])},{int(tpPointsChoose[7][1])}'\n"
      f" enter_south ='{int(tpPointsChoose[8][0])},{int(tpPointsChoose[8][1])},{int(tpPointsChoose[9][0])},{int(tpPointsChoose[9][1])},{int(tpPointsChoose[10][0])},{int(tpPointsChoose[10][1])},{int(tpPointsChoose[11][0])},{int(tpPointsChoose[11][1])}'\n"
      f" enter_north ='{int(tpPointsChoose[12][0])},{int(tpPointsChoose[12][1])},{int(tpPointsChoose[13][0])},{int(tpPointsChoose[13][1])},{int(tpPointsChoose[14][0])},{int(tpPointsChoose[14][1])},{int(tpPointsChoose[15][0])},{int(tpPointsChoose[15][1])}'\n"
      f" exit_east ='{int(tpPointsChoose[16][0])},{int(tpPointsChoose[16][1])},{int(tpPointsChoose[17][0])},{int(tpPointsChoose[17][1])},{int(tpPointsChoose[18][0])},{int(tpPointsChoose[18][1])},{int(tpPointsChoose[19][0])},{int(tpPointsChoose[19][1])}'\n"
      f" exit_west ='{int(tpPointsChoose[20][0])},{int(tpPointsChoose[20][1])},{int(tpPointsChoose[21][0])},{int(tpPointsChoose[21][1])},{int(tpPointsChoose[22][0])},{int(tpPointsChoose[22][1])},{int(tpPointsChoose[23][0])},{int(tpPointsChoose[23][1])}'\n"
      f" exit_south ='{int(tpPointsChoose[24][0])},{int(tpPointsChoose[24][1])},{int(tpPointsChoose[25][0])},{int(tpPointsChoose[25][1])},{int(tpPointsChoose[26][0])},{int(tpPointsChoose[26][1])},{int(tpPointsChoose[27][0])},{int(tpPointsChoose[27][1])}'\n"
      f" exit_north ='{int(tpPointsChoose[28][0])},{int(tpPointsChoose[28][1])},{int(tpPointsChoose[29][0])},{int(tpPointsChoose[29][1])},{int(tpPointsChoose[30][0])},{int(tpPointsChoose[30][1])},{int(tpPointsChoose[31][0])},{int(tpPointsChoose[31][1])}'\n"
      )