'''
BRODY v0.1 - Exclude_outlying module

'''

from natsort import natsorted
import os
import cv2
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt

def Exclude_boundary_instance(img_name, extream_point_list, filtered_index, z_c_list):
    """ 이미지 경계에 위치하여 잘린 개체를 배제하는 함수.  
    Args:
        img_name: RGB 파일 디렉토리 경로 
        extream_point_list: 모든 개체의 상하좌우 극점의 2차원 좌표가 저장된 리스트
        filtered_index: 예외처리된 모든 개체의 인덱스가 저장된 리스트
        z_c_list: 모든 개체의 무게중심점까지의 z값이 저장된 리스트

    Returns:
        exclude_boundary_index_list: 경계에서 잘린 개체를 제외한 나머지 개체의 인덱스가 저장된 리스트
        exclude_boundary_z_c_list: 경계에서 잘린 개체를 제외한 나머지 개체의 무게중심점 z값이 저장된 리스트
    """
    # 입력 이미지 사이즈 반환.
    img_arr = cv2.imread(img_name)
    height, width, channel = img_arr.shape

    # 최외곽선 선언
    line1 = cv2.line(img_arr, (0,0), (height,0), color = (0,0,0), thickness = 1)
    line2 = cv2.line(img_arr, (0,0), (0,width), color = (0,0,0), thickness = 1)
    line3 = cv2.line(img_arr, (height,0), (height,width), color = (0,0,0), thickness = 1)
    line4 = cv2.line(img_arr, (0,width), (height,width), color = (0,0,0), thickness = 1)

    # offset 선언
    offset = int(height* 0.01)
    outlying_list = []

    # offset을 적용한 최외곽선과 만나거나 외곽에 위치한 개체 배제
    for index,points in enumerate(extream_point_list):
        if points[0][1] <= offset: # topmost의 y좌표
            if index in filtered_index:
                filtered_index.remove(index)
            else:
                pass

        elif points[1][1] >= height - offset: # bottommost의 y좌표
            if index in filtered_index:
                filtered_index.remove(index)
            else:
                pass

        elif points[2][0] <= offset: # leftmost의 x좌표
            if index in filtered_index:
                filtered_index.remove(index)
            else:
                pass

        elif points[3][0] >= width - offset: # rightmost x좌표 
            if index in filtered_index:
                filtered_index.remove(index)
            else:
                pass

        else: 
            pass
    
    exclude_boundary_index_list = filtered_index
    exclude_boundary_z_c_list = []
    for i in exclude_boundary_index_list:
        exclude_boundary_z_c_list.append(z_c_list[i])
    
    return exclude_boundary_index_list, exclude_boundary_z_c_list

def Exclude_depth_error(exclude_boundary_index_list, exclude_boundary_z_c_list, mask_list):
    """ 깊이 이상치를 갖는 개체를 배제하는 함수.  
    Args:
        exclude_boundary_index_list: 이미지 경계에 위치한 개체를 제외한 나머지 개체 인덱스가 저장된 리스트
        exclude_boundary_z_c_list: 이미지 경계에 위치한 개체를 제외한 나머지 개체 무게중심점 z값이 저장된 리스트
        mask_list: 모든 개체의 mask 정보가 저장된 리스트

    Returns:
        exclude_depth_err_index_list: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트
        exclude_depth_err_mask_list: 깊이 이상치 개체를 제외한 나머지 개체 mask 정보 저장된 리스트
    """
    # 깊이 이상치를 갖는 개체 제거
    exclude_boundary_z_c_list = list(map(int,exclude_boundary_z_c_list))
    mean = sum(exclude_boundary_z_c_list)/len(exclude_boundary_z_c_list)
    
    exclude_depth_err_index_list = []
    for index, value in enumerate(exclude_boundary_z_c_list):
        if value > mean*1.2 or value < mean*0.8:
            print("----- ",exclude_boundary_index_list[index], "번 개체가 배제되었습니다. -----")
            pass
        else:
            exclude_depth_err_index_list.append(exclude_boundary_index_list[index]) # 깊이정보가 올바른 개체
    
    # mask list에서 깊이 이상치를 갖는 개체 제거
    exclude_depth_err_mask_list = []
    for i in exclude_depth_err_index_list:
        exclude_depth_err_mask_list.append(mask_list[i])
    
    return exclude_depth_err_index_list, exclude_depth_err_mask_list