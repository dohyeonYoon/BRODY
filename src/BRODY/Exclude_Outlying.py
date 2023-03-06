'''
BRODY v0.1 - Exclude_outlying module

'''

import cv2

def Exclude_boundary_instance(img_name, extream_point_list, filtered_index):
    """ 이미지 경계에 위치하여 잘린 개체를 배제하는 함수.  
    Args:
        img_name: RGB 파일 디렉토리 경로 
        extream_point_list: 모든 개체의 상하좌우 극점의 2차원 좌표가 저장된 리스트
        filtered_index: 예외처리된 모든 개체의 인덱스가 저장된 리스트

    Returns:
        exclude_boundary_index_list: 경계에서 잘린 개체를 제외한 나머지 개체의 인덱스가 저장된 리스트
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
    
    return exclude_boundary_index_list


def Exclude_depth_error(z_c_list, exclude_boundary_index_list):
    """ 깊이 이상치를 갖는 개체를 배제하는 함수.  
    Args:
        z_c_list: 모든 개체의 무게중심점에 median filter가 적용된 z값을 저장한 리스트
        exclude_boundary_index_list: 이미지 경계에 위치한 개체를 제외한 나머지 개체 인덱스가 저장된 리스트

    Returns:
        exclude_depth_err_index_list: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트
    """
    # 깊이 이상치를 갖는 개체 제거
    z_c_list = list(map(int,z_c_list))
    mean = sum(z_c_list)/len(z_c_list)

    for index, value in enumerate(z_c_list):
        if value > mean*1.2 or value < mean*0.8:
            if index in exclude_boundary_index_list:
                exclude_boundary_index_list.remove(index) # 깊이 이상치 제거
                print(f"-----{index}번 개체가 배제되었습니다. -----")
            else: 
                print(f"-----{index}번 개체는 이미 배제되었습니다. -----")
            
        else:
            pass
    
    exclude_depth_err_index_list = exclude_boundary_index_list

    return exclude_depth_err_index_list


def Find_straight_line(contour_list):

    # 개별 인스턴스가 길이 35pixel 이상의 직선을 가지고 있는지 확인.
    for contour in contour_list:
        for j in range(len(contour[0])-1):
            # Calculate the distance between two adjacent contour points
            dist = cv2.norm(contour[0][j], contour[0][j+1])
            if dist >= 35:
                print(f"There is a straight line of 35 pixels or more in the contour in 번째 instace.")
                break
    return 