'''
BRODY v0.1 - Prediction module

'''

from shapely.geometry import Polygon
import numpy as np 
import joblib

def Calculate_2d_area(contour_list_3d, exclude_depth_err_index_list):
    """3차원 contour점으로 이루어진 다각형의 면적 계산해주는 함수.

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
    print(average_area)
    average_perimeter = round(sum(perimeter_list,0.0)/ len(perimeter_list),2)

    return area_list, average_area, perimeter_list, average_perimeter


def Calculate_weight(area_list, perimeter_list, major_axis_list, minor_axis_list):
    """회귀방정식을 통해 면적으로부터 체중을 예측해주는 함수.

    Args:
        area_list: 각 instance의 면적이 저장된 list.
        perimeter_list: 모든 개체의 둘레가 저장된 리스트 
        major_axis_list: 모든 개체의 major_axis 길이가 저장된 리스트
        minor_axis_list: 모든 개체의 minor_axis 길이가 저장된 리스트

    Returns:
        predict_weight_list: 각 instance의 예측체중이 저장된 list.
    """
    weight_list = []
    model = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/Huber.pkl')# 기존 면적-체중 선형회귀모델
    # model1 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/model1.pkl') # 면적-체중 다중선형회귀모델
    # model2 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/model2.pkl') # 일령,면적-체중 다중선형회귀모델
    # model3 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/model3.pkl') # 일령,면적,둘레-체중 다중선형회귀모델
    # model4 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/model4.pkl') # 일령,면적,둘레,장축-체중 다중선형회귀모델
    # norm_model1 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/norm_model1.pkl') # 면적-체중 다중선형회귀모델(정규화)
    # norm_model2 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/norm_model2.pkl') # 일령,면적-체중 다중선형회귀모델(정규화)
    # norm_model3 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/norm_model3.pkl') # 일령,면적,둘레-체중 다중선형회귀모델(정규화)
    # norm_model4 = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/norm_model4.pkl') # 일령,면적,둘레,장축-체중 다중선형회귀모델(정규화)
    # polynomial_model = joblib.load('/scratch/dohyeon/BRODY/src/method_override/linear_model/polynomiar_regression_model.pkl')

    # days_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    # scaled_days_list, scaled_area_list, scaled_perimeter_list, scaled_major_axis_list = Min_max_scale(days_list, area_list, perimeter_list, major_axis_list)


    # 정규화 x 
    for i in range(len(area_list)):
        # day = days_list[days-1]
        area = area_list[i]
        # perimeter = perimeter_list[i]
        # major_axis = major_axis_list[i]
        weight = model.predict([[area]]) # 면적-체중 모델
        weight_list.append(weight)
    
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

    #     weight = model1 # 일령, 면적, 둘레, 장축, - 체중 모델
    #     # weight = model4.predict([[day, area, perimeter, major_axis]]) # 일령, 면적, 둘레, 장축, - 체중 모델
    #     weight_list.append(weight)


    # 예측 평균체중, 마리수 변수에 저장.
    average_weight = round((sum(weight_list,0.0)/ len(weight_list))[0], 2)
    print("추론된 육계군집의 예측 평균체중은 ",average_weight,"g 입니다.")

    return weight_list, average_weight


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