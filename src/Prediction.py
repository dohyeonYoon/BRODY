'''
BRODY v0.1 - Prediction module

'''

from shapely.geometry import Polygon
import numpy as np 
import joblib

def Calculate_2D_Area(contour_list, array_3d, th_index):
    """3차원 contour점으로 이루어진 다각형의 면적 계산해주는 함수.

    Args:
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
        th_index: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트

    Returns:
        area_list: 모든 개체의 면적이 저장된 리스트
    """
    # Calculate 3d contour points.
    contour_list_3d = []
    for instance in contour_list:
        instnace_points = []
        # for point in instance[0]:
        for point in instance:
            y,x = point[0][1], point[0][0]
            xyz = array_3d[y,x]
            instnace_points.append(xyz)
        contour_list_3d.append(instnace_points)
    contour_list_3d = np.array(contour_list_3d, dtype= object)

    # Calculate area of all objects.
    area_list = []
    for i in th_index:
        polygon = Polygon(np.array(contour_list_3d[i]))
        polygon_area = round((polygon.area)/100,2)
        area_list.append(polygon_area)

    return area_list


def Calculate_Weight(area_list):
    """회귀방정식을 통해 면적으로부터 체중을 예측해주는 함수.

    Args:
        area_list: 모든 개체의 면적이 저장된 리스트

    Returns:
        weight_list: 모든 개체의 예측체중이 저장된 리스트
    """
    # Load linear regression model.
    model = joblib.load('./regression/weights/Huber.pkl')

    # Calculate weight of all objects.
    weight_list = []
    for i in range(len(area_list)):
        area = area_list[i]
        weight = model.predict([[area]])
        weight_list.append(weight)

    return weight_list