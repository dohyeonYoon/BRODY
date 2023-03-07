'''
BRODY v0.1 - Convert module

'''

import numpy as np
from shapely.geometry import Polygon, mapping, shape
import joblib
import cv2
from math import sqrt

def Convert_2D_to_3D(img_name, depthmap_name, contour_list, centroid_list):
    """2차원 픽셀좌표계를 3차원 월드좌표계로 변환해주는 함수.

    Args:
        img_name: RGB(.png) 파일 디렉토리 경로
        depthmap_name: Depthmap(.pgm) 파일 디렉토리 경로
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        centroid_list (ndarray): 모든 개체의 무게중심점 픽셀좌표가 저장된 리스트

    Returns:
        contour_list_3d: 모든 개체의 contour점 3D-월드좌표가 저장된 리스트
        z_c_list: 모든 개체의 무게중심점에 median filter가 적용된 z값을 저장한 리스트
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
    """
    # 입력 이미지 사이즈 반환.
    height, width, channel = cv2.imread(img_name).shape

    # Depth 정보 parsing.
    depth_list = []
    with open(depthmap_name, 'r') as f:
        data = f.readlines()[3:]
        for i in data:
            for j in i.split():
                depth_list.append(int(j))

    # depth map을 이미지 형태(height*width)로 reshape.
    depth_map = np.array(depth_list)
    depth_map = np.reshape(depth_map, (height,width))

    # 무게중심점에 3x3 size median filter 적용
    windows = [0,0,0,0,0,0,0,0,0] # 3x3 window
    median_array= depth_map.copy()
    for i in centroid_list:
        windows[0] = depth_map[i[1]-1, i[0]-1] # height, width 순서
        windows[1] = depth_map[i[1]-1, i[0]]
        windows[2] = depth_map[i[1]-1, i[0]+1]
        windows[3] = depth_map[i[1], i[0]-1]
        windows[4] = depth_map[i[1], i[0]]
        windows[5] = depth_map[i[1], i[0]+1]
        windows[6] = depth_map[i[1]+1, i[0]-1]
        windows[7] = depth_map[i[1]+1, i[0]]
        windows[8] = depth_map[i[1]+1, i[0]+1]
        windows = np.array(windows)
        windows.sort()
        median_array[i[1], i[0]] = windows[4]

    # median filter가 적용된 무게중심점의 z값을 contour에 뿌리기
    z_c_list = []
    for j in range(len(centroid_list)):
        z_c_list.append(median_array[centroid_list[j][1],centroid_list[j][0]])
        for k in contour_list[j][0]:
            median_array[k[0][1],k[0][0]] = int(median_array[centroid_list[j][1],centroid_list[j][0]])

    # 카메라 내부 파라미터를 이용한 3차원 월드좌표(실제좌표)로 변환.
    fx = 535.14 # 초점거리 x
    fy = 535.325 # 초점거리 y
    cx= 646.415 # 주점 x
    cy= 361.3215 # 주점 y
    factor = 1.0

    # array_3d 리스트에 X,Y,Z 좌표 저장
    array_3d = []
    for u in range(height):
        for v in range(width):
            Z = float(median_array[u,v]) / factor # 해당 픽셀의 3차원상의 실제좌표 z
            Y= ((u-cy) * float(Z)) / fy # 해당 픽셀의 3차원상의 실제좌표 y
            X= ((v-cx) * float(Z)) / fx # 해당 픽셀의 3차원상의 실제좌표 x
            array_3d.append([X,Y,Z])
    array_3d = np.array(array_3d)
    array_3d = array_3d.reshape(height,width,3)

    # 각 instance의 contour points 3차원 좌표 list에 저장.
    contour_list_3d = []
    for i in range(len(contour_list)):
        point_list0 = []
        for j in range(len(contour_list[i][0])): # i번째 육계의 contour 개수
            points = array_3d[contour_list[i][0][j][0][1], contour_list[i][0][j][0][0]] # height, width 순서 
            point_list0.append(points)
        contour_list_3d.append(point_list0)
    contour_list_3d = np.array(contour_list_3d, dtype= object)

    return z_c_list, array_3d, contour_list_3d

def Calculate_major_minor_axis(extream_point_list, array_3d):
    """개별 개체의 장축,단축 길이 계산해주는 함수.

    Args:
        extream_point_list: 모든 개체의 상하좌우 극점이 저장된 리스트
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array

    Returns:
        major_axis_list: 모든 개체의 major_axis 길이가 저장된 리스트
        minor_axis_list: 모든 개체의 minor_axis 길이가 저장된 리스트
    """
    # extream point를 잇는 major, minor axis length 계산
    major_axis_list = []
    minor_axis_list = []

    for i in range(len(extream_point_list)):
        tm_2d = list(extream_point_list[i][0]) # 상 2D
        bm_2d = list(extream_point_list[i][1]) # 하 2D
        lm_2d = list(extream_point_list[i][2]) # 좌 2D
        rm_2d = list(extream_point_list[i][3]) # 우 2D

        # 상하좌우 극점의 3D 좌표
        tm_3d = list(array_3d[tm_2d[1], tm_2d[0]]) # 상 3D
        bm_3d = list(array_3d[bm_2d[1], bm_2d[0]]) # 하 3D
        lm_3d = list(array_3d[lm_2d[1], lm_2d[0]]) # 좌 3D
        rm_3d = list(array_3d[rm_2d[1], rm_2d[0]]) # 우 3D

        # 상-하, 좌-우 3D 극점 사이의 거리
        distance_tb = sqrt((tm_3d[0]-bm_3d[0])**2 + (tm_3d[1]-bm_3d[1])**2 + (tm_3d[2]-bm_3d[2])**2)
        distance_lr = sqrt((lm_3d[0]-rm_3d[0])**2 + (lm_3d[1]-rm_3d[1])**2 + (lm_3d[2]-rm_3d[2])**2) 

        # 더 큰 값을 major_axis_list에 삽입
        bigger_distance = round(max(distance_tb, distance_lr),2)
        smaller_distnace = round(min(distance_tb, distance_lr),2)
        major_axis_list.append(bigger_distance)
        minor_axis_list.append(smaller_distnace)

    return major_axis_list, minor_axis_list

def Calculate_perpendicular_point(array_3d):
    w,h = array_3d.shape[0], array_3d.shape[1]
    total = w*h 
    array3 = array_3d.reshape(total,3)

    # 포인트 클라우드 정의
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(array3)
    pcd_plane.normals = o3d.utility.Vector3dVector(array3)

    # 지면 법선벡터 계산 및 flip 
    pcd_plane.estimate_normals()
    pcd_plane.orient_normals_towards_camera_location(pcd_plane.get_center())
    pcd_plane.normals = o3d.utility.Vector3dVector( - np.asarray(pcd_plane.normals))

    # 지면 평면 계산 및 시각화
    plane_model, inliers = pcd_plane.segment_plane(distance_threshold=0.1,
                                        ransac_n=3,
                                        num_iterations=1000)
    [a, b, c, camera_height] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {camera_height:.2f} = 0")  
    camera_height = - camera_height

    perpendicular_point = (0,0,camera_height)

    return perpendicular_point

def Calculate_distance(array_3d, centroid_list, perpendicular_point, exclude_index_list):
    """각 instance의 무게중심점의 3차원 좌표를 추출하고, 수선의발에서부터 각 instance까지 거리 구하는 함수.

    """

    # 각 instance의 무게중심점 3차원 좌표 list에 저장.
    center_of_mass_list = []
    exclude_centroid_list = []

    for i in exclude_index_list:
        exclude_centroid_list.append(centroid_list[i])

    for i in range(len(exclude_centroid_list)): # 육계 개체 수
        point_x, point_y = array_3d[exclude_centroid_list[i][1], exclude_centroid_list[i][0]][0],array_3d[exclude_centroid_list[i][1], exclude_centroid_list[i][0]][1]  # height, width 순서 
        center_of_mass_list.append([point_x,point_y])
    
    # 유클리디안 거리 계산
    perpendicular_point = [0,0]
    distance_list = []

    for i in center_of_mass_list:
        distance = sqrt((perpendicular_point[0]-i[0])**2 +(perpendicular_point[1]-i[1])**2)
        distance_list.append(distance)
    
    return distance_list