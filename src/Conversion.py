'''
BRODY v0.1 - Convert module

'''

import numpy as np
from shapely.geometry import Polygon, mapping, shape
import joblib
import cv2
from math import sqrt


def Generate_Depthmap_1(filename, contour_list, th_index):
    """깊이정보를 받아와서 각 개체의 contour에 3x3 median filter를 적용해주는 함수.

    Args:
        filename: 입력 데이터 이름
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트

    Returns:
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
    """
    # Set image,depthmap file name.
    img_name = filename + '.png'
    depth_name = filename + '.pgm'

    # Return input image size.
    height, width, channel = cv2.imread(img_name).shape

    # Depth data parsing.
    depth_list = []
    with open(depth_name, 'r') as f:
        lines = f.readlines()
        if lines[0] == 'P2\n' and lines[1] == f'{width} {height}\n':

            # for i in lines[3:]:
            #     for j in i.split():
            #         depth_list.append(int(j))
            depth_list = [int(j) for line in lines[3:] for j in line.split()]
        else:
            raise ValueError("depthmap file이 pgm ascii 형식이 아니거나 RGB 이미지 크기와 다릅니다.")

    # Reshape depthmap into image shape.
    depth_map = np.array(depth_list)
    depth_map = np.reshape(depth_map, (height,width))

    # # Apply 3x3 size median filter to contour points.
    # for i in contour_list:
    #     for j in i:
    #         depth_map[j[0][1], j[0][0]] = np.median(depth_map[j[0][1]-1:j[0][1]+2, j[0][0]-1:j[0][0]+2])

    return depth_map


def Convert_3D(filename, depth_map, mask_list):
    """2차원 픽셀좌표계를 3차원 월드좌표계로 변환해주는 함수.

    Args:
        filename: 입력 데이터 이름
        depth_map: 모든 픽셀의 깊이정보(z값)가 저장된 array

    Returns:
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
    """
    # Set image,depthmap file name.
    img_name = filename + '.png'

    # Return input image size.
    height, width, channel = cv2.imread(img_name).shape

    # Convert to 3D world coordinates using camera internal parameters
    fx = 535.14 # focal length x
    fy = 535.325 # focal length y
    cx= 646.415 # principal point x
    cy= 361.3215 # principal point y

    # Convert depth value to 3D point cloud
    u, v = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    Z = depth_map
    X = (v - cx) * Z / fx
    Y = (u - cy) * Z / fy

    array_3d = np.dstack((X, Y, Z))

    # Generate 3D coordinates for the mask list
    mask_list_3d = []
    for mask in mask_list:
        points = [array_3d[pt[0][1], pt[0][0]] for pt in mask]
        mask_list_3d.append(points)

    return array_3d, mask_list_3d