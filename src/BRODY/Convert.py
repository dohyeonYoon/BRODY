'''
BRODY v0.1 - Convert module

'''

import numpy as np
import os
from shapely.geometry import Polygon, mapping, shape
from natsort import natsorted
from csv import writer
from datetime import datetime
import joblib
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
import time
import pickle
import random
import open3d as o3d
import alphashape
from sklearn.preprocessing import MinMaxScaler
import time
from math import sqrt

def Calculate_day(rgb_file_name, arrival_date):
    """현재날짜 육계의 일령 계산해주는 함수.

    Args:
        rgb_file_name: 이미지 파일 경로
        arrival_date: 병아리가 들어온 일자(1일령일 때 날짜)

    Returns:
        days: 현재날짜의 육계군집의 일령(육계의 나이는 1일령,5일령...등으로 표현).
    """
    # 파일 이름으로부터 날짜 추출.
    date = os.path.splitext(rgb_file_name)[0]
    date_value = date.split('_')
    date_value = date_value[0:]

    # 촬영날짜 및 병아리 입식날짜
    dt1 = datetime(int(date_value[0]),int(date_value[1]),int(date_value[2]),int(date_value[3]))
    dt2 = datetime(arrival_date[0],arrival_date[1],arrival_date[2],arrival_date[3]) # 육계가 1일령일 때 날짜를 입력해줄 것(시간은 00시 기준).

    # 일령계산
    td = dt1-dt2
    days = td.days + 1

    return days

def Convert_2D_to_3D_area(img_name, depthmap_name, contour_list, centroid_list):
    """depth map(.pgm) file을 불러와서 개별 개체의 contour point들의 z값을 무게중심점의 z값으로 대체한 뒤, 3차원 월드좌표로 변환해주는 함수.

    Args:
        img_name: RGB 파일 디렉토리 경로
        depthmap_name: Depthmap(.pgm) 파일 디렉토리 경로
        contour_list (ndarray): 각 instance의 contour점 픽셀좌표 리스트
        centroid_list (ndarray): 각 instance의 무게중심점 픽셀좌표 리스트

    Returns:
        contour_list_3d: 모든 개체의 3차원 contour point들이 저장된 리스트
        z_c_list: 모든 개체의 무게중심점의 z값을 저장한 리스트
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
    """
    # 입력 이미지 사이즈 반환.
    height, width, channel = cv2.imread(img_name).shape

    # Depth data parsing.
    depth_list = []
    with open(depthmap_name, 'r') as f:
        data = f.readlines()[3:]
        for i in data:
            for j in i.split():
                depth_list.append(int(j))

    # depth map을 이미지 사이즈(height*width) 형태로 reshape.
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
    array_3d = array_3d.reshape(720,1280,3)

    # 각 instance의 contour points 3차원 좌표 list에 저장.
    contour_list_3d = []
    for i in range(len(contour_list)): # 예외처리를 한 contour 배열의 원수 개수 = 제대로 segmentation된 개체 수
        point_list0 = []
        for j in range(len(contour_list[i][0])): # i번째 육계의 contour 개수
            points = array_3d[contour_list[i][0][j][0][1], contour_list[i][0][j][0][0]] # height, width 순서 
            point_list0.append(points)
        contour_list_3d.append(point_list0)
    contour_list_3d = np.array(contour_list_3d, dtype= object)

    return contour_list_3d, z_c_list, array_3d

def Calculate_major_minor_axis(extream_point_list, array_3d):
    """개별 인스턴스의 극점의 좌표를 받아와서 major, minor axis 길이 계산해주는 함수.

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

        # 좌우상하 극점의 3D 좌표
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

def Calculate_2d_area(contour_list_3d, exclude_depth_err_index_list):
    """입력받은 3차원 리스트로 polygon을 형성한 뒤, 면적을 계산해주는 함수.

    Args:
        contour_list_3d: 모든 개체의 3차원 contour점이 저장된 리스트
        exclude_depth_err_index_list: 깊이 이상치 개체를 배제한 나머지 개체들의 인덱스가 저장된 리스트

    Returns:
        area_list: 모든 개체의 면적이 저장된 리스트
        average_area: 모든 개체의 평균면적
        perimeter_list: 모든 개체의 둘레가 저장된 리스트 
        average_perimeter: 모든 개체의 평균둘레
    """
    # 면적 및 둘레 계산
    area_list = []
    perimeter_list = []
    for i in exclude_depth_err_index_list:
        polygon = Polygon(np.array(contour_list_3d[i]))
        polygon_area = round((polygon.area)/100,2)
        area_list.append(polygon_area)
        polygon_perimeter = round((polygon.length)/10,2)
        perimeter_list.append(polygon_perimeter)
    
    # 평균면적 및 평균둘레 계산
    average_area = round(sum(area_list,0.0)/ len(area_list),2)
    average_perimeter = round(sum(perimeter_list,0.0)/ len(perimeter_list),2)

    return area_list, average_area, perimeter_list, average_perimeter

def Calculate_perpendicular_point(array2):
    w,h = array2.shape[0], array2.shape[1]
    total = w*h 
    array3 = array2.reshape(total,3)

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
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {camera_height:.2f} = 0")  
    camera_height = - camera_height

    perpendicular_point = (0,0,camera_height)

    return perpendicular_point

def Calculate_distance(array2, centroid_list, perpendicular_point, exclude_index_list):
    """각 instance의 무게중심점의 3차원 좌표를 추출하고, 수선의발에서부터 각 instance까지 거리 구하는 함수.

    """

    # 각 instance의 무게중심점 3차원 좌표 list에 저장.
    center_of_mass_list = []
    exclude_centroid_list = []

    for i in exclude_index_list:
        exclude_centroid_list.append(centroid_list[i])

    for i in range(len(exclude_centroid_list)): # 육계 개체 수
        point_x, point_y = array2[exclude_centroid_list[i][1], exclude_centroid_list[i][0]][0],array2[exclude_centroid_list[i][1], exclude_centroid_list[i][0]][1]  # height, width 순서 
        center_of_mass_list.append([point_x,point_y])
    
    # 유클리디안 거리 계산
    perpendicular_point = [0,0]
    distance_list = []

    for i in center_of_mass_list:
        distance = sqrt((perpendicular_point[0]-i[0])**2 +(perpendicular_point[1]-i[1])**2)
        distance_list.append(distance)
    
    return distance_list

def Min_max_scale(days_list, area_list, perimeter_list, major_axis_list):
    max_day = 35
    max_area = 404.62
    max_perimeter = 106.15
    max_major_axis = 347.39 
    min_day = 1
    min_area = 28.71
    min_perimeter = 20.94
    min_major_axis = 66.61

    scaled_days_list = []
    scaled_area_list = []
    scaled_perimeter_list = []
    scaled_major_axis_list = []

    for i in days_list:
        new_days_x = (i - min_day) / (max_day - min_day)
        scaled_days_list.append(new_days_x)

    for j in area_list:
        new_area_x = (j - min_area) / (max_area - min_area)
        scaled_area_list.append(new_area_x)

    for k in perimeter_list:
        new_perimeter_x = (k - min_perimeter) / (max_perimeter - min_perimeter)
        scaled_perimeter_list.append(new_perimeter_x)

    for m in perimeter_list:
        new_major_axis_x = (m - min_major_axis) / (max_major_axis - min_major_axis)
        scaled_major_axis_list.append(new_major_axis_x)


    return scaled_days_list, scaled_area_list, scaled_perimeter_list, scaled_major_axis_list


def Calculate_weight(exclude_index_list, days, area_list, perimeter_list, major_axis_list):
    """면적 list를 입력으로 받아 회귀방정식을 통해 체중을 예측해주는 함수.

    Args:
        area_list: 각 instance의 면적이 저장된 list.
        inner_object_list: 최외곽선 내부에 위치한 개체 list

    Returns:
        predict_weight_list: 각 instance의 예측체중이 저장된 list.
    """

    predict_weight_list = []
    linear_model = joblib.load('/scratch/dohyeon/mmdetection/linear_model/linear.pkl')
    model = joblib.load('/scratch/dohyeon/mmdetection/linear_model/Huber.pkl')# 기존 면적-체중 선형회귀모델
    model1 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/model1.pkl') # 면적-체중 다중선형회귀모델
    model2 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/model2.pkl') # 일령,면적-체중 다중선형회귀모델
    model3 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/model3.pkl') # 일령,면적,둘레-체중 다중선형회귀모델
    model4 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/model4.pkl') # 일령,면적,둘레,장축-체중 다중선형회귀모델
    norm_model1 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/norm_model1.pkl') # 면적-체중 다중선형회귀모델(정규화)
    norm_model2 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/norm_model2.pkl') # 일령,면적-체중 다중선형회귀모델(정규화)
    norm_model3 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/norm_model3.pkl') # 일령,면적,둘레-체중 다중선형회귀모델(정규화)
    norm_model4 = joblib.load('/scratch/dohyeon/mmdetection/linear_model/norm_model4.pkl') # 일령,면적,둘레,장축-체중 다중선형회귀모델(정규화)
    polynomial_model = joblib.load('/scratch/dohyeon/mmdetection/linear_model/polynomiar_regression_model.pkl')

    days_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    scaled_days_list, scaled_area_list, scaled_perimeter_list, scaled_major_axis_list = min_max_feature(days_list, area_list, perimeter_list, major_axis_list)


    # 정규화 x 
    for i in range(len(area_list)):
        day = days_list[days-1]
        area = area_list[i]
        perimeter = perimeter_list[i]
        major_axis = major_axis_list[i]
        predict_weight = model.predict([[area]]) # 면적-체중 모델
        predict_weight_list.append(predict_weight)
    
    # # 정규화 O
    # for i in range(len(area_list)):
    #     day = scaled_days_list[days-1]
    #     area = scaled_area_list[i]
    #     perimeter = scaled_perimeter_list[i]
    #     major_axis = scaled_major_axis_list[i]
    #     # print(f"일령, 면적, 둘레, 장축 {day}, {area}, {perimeter}, {major_axis}")
        
    #     model1 = 5.2594+ 2154.6709*area + 597.038*area*area 
    #     model2 = 15.9082+ 832.9550*day + 676.0475*day*day + 233.4223*area + 616.6395*area*area
    #     model3 = 22.0761+ 785.6337*day + 666.0536*day*day + 228.9946*area + 677.8528*area*area -18.2471*perimeter -23.8522*perimeter*perimeter 
    #     model4 = 24.9496+ 781.6776*day + 662.5781*day*day + 244.1271*area + 669.8963*area*area -34.2661*perimeter -40.9845*perimeter*perimeter -46.0709*major_axis +79.0229*major_axis*major_axis

    #     predict_weight = model1 # 일령, 면적, 둘레, 장축, - 체중 모델
    #     # print(predict_weight)
    #     # predict_weight = model4.predict([[day, area, perimeter, major_axis]]) # 일령, 면적, 둘레, 장축, - 체중 모델
    #     predict_weight_list.append(predict_weight)


    # 예측 평균체중, 마리수 변수에 저장.
    predict_average_weight = int((sum(predict_weight_list,0.0)/ len(predict_weight_list)))

    num_chicken = len(predict_weight_list)
    
    print("추론된 육계군집의 예측 평균체중은 ",predict_average_weight,"g 입니다.")

    return predict_weight_list, predict_average_weight, num_chicken

def Save_to_csv1(days, predict_average_weight):
    """날짜, 탐지된 육계 개체수(마리), 실제면적, 예측체중을 csv file에 저장해주는 함수.

    Args:
        date: 영상이 촬영된 날짜.
        days: 일령.
        area_list: 각 instance의 면적이 저장된 list.
        predict_weight_list: 각 instance의 예측체중이 저장된 list.

    Returns:
        None
    """
    # 촬영날짜, 일령, 면적, 체중 4가지를 csv file에 저장.
    rows = [days, predict_average_weight]
    with open(f'/scratch/dohyeon/mmdetection/output/avg_weight_result.csv','a', newline='') as f_object:
        # using csv.writer method from CSV package
        writer_object = writer(f_object)
        writer_object.writerow(rows)
        f_object.close()
    return

def Visualize_weight(input_path,
                    output_path,
                    results, 
                    predict_weight_list, 
                    exclude_index_list, 
                    predict_average_weight, 
                    days,
                    num_chicken,
                    real_z_c_list,
                    average_area, 
                    score_threshold,
                    area_list,
                    rgb_file_name,):
    """각 개체마다 id number, 예측체중 시각화해주는 함수.

    Args:
        input_path: 입력 이미지를 불러오는 디렉토리 경로.
        output_path: 입력 이미지 위에 segmentation 결과가 그려진 이미지가 저장되는 디렉토리 경로
        reesults: segment_chicken 함수의 반환값(Instance segmentation 결과값 배열)
        area_list: 예측면적 저장된 리스트
        date: 파일이름에서 추출한 촬영시각
        z_c_list: 모든 개체의 무게중심점의 z값을 저장한 list
        
    Returns:
        None
    """
    
    # config 파일을 설정하고, 학습한 checkpoint file 불러오기.
    # config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_15.py' # Mask-RCNN-Dataset_15
    # checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_15/epoch_36.pth' # Mask-RCNN-Dataset_15
    config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_30.py' # Mask-RCNN-Dataset_30
    checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_30/epoch_36.pth' # Mask-RCNN-Dataset_30
    # config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_68.py' # Mask-RCNN-Dataset_68
    # checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_68/epoch_35.pth' # Mask-RCNN-Dataset_68
    # config_file = '/scratch/dohyeon/mmdetection/custom_config/num_dataset_87.py' # Mask-RCNN-Dataset_87
    # checkpoint_file = '/scratch/dohyeon/mmdetection/weights/mask_rcnn_r101/num_dataset_87/epoch_35.pth' # Mask-RCNN-Dataset_87

    # config 파일과 checkpoint를 기반으로 Detector 모델을 생성.
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # 경로선언.
    path_dir = input_path
    save_dir1 = output_path
    file_list = natsorted(os.listdir(path_dir))
    
    # 입력 이미지에 추론결과 visualize
    for i in range(1): 
        img_name = path_dir + '/' + rgb_file_name
        img_arr= cv2.imread(img_name, cv2.IMREAD_COLOR)
        img_arr_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        fig= plt.figure(figsize=(12, 12))
        plt.imshow(img_arr_rgb)

        # 추론결과 디렉토리에 저장(confidenece score 0.7이상의 instance만 이미지에 그릴 것).
        model.show_result(img_arr,
                        results,
                        predict_weight_list,
                        days,
                        num_chicken,
                        average_area,
                        exclude_index_list,
                        predict_average_weight,
                        area_list,
                        real_z_c_list,
                        score_thr=score_threshold,
                        bbox_color=(0,0,0),
                        thickness=0.01,
                        font_size=8,
                        out_file= f'{save_dir1}{rgb_file_name}')
    return

def Empty_dir(file_path):
    """디렉토리 내의 모든 파일 삭제해주는 함수.

    Args:
        file_path: 디렉토리 경로.
        
    Returns:
        'Remove All File': 지정된 디렉토리 내의 삭제할 파일이 존재하면 모든 파일을 삭제
        'Directory Not Found': 지정된 디렉토리를 찾을 수 없으면 "Directory Not Found" 출력
    """
    if os.path.exists(file_path):
        for file in os.scandir(file_path):
            os.remove(file.path)
        return 'Remove All File'
    else:
        return 'Directory Not Found'


        