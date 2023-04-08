'''
BRODY v0.1 - Exclusion module

'''

import cv2
import numpy as np

def Delete_Exterior(filename, contour_list, th_index):
    """ 이미지 경계에 위치하여 잘린 개체를 배제해주는 함수.  
    Args:
        filename: 입력 데이터 이름
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        th_index: 예외처리 후 나머지 개체의 인덱스가 저장된 리스트

    Returns:
        th_index: 경계에서 잘린 개체를 제외한 나머지 개체의 인덱스가 저장된 리스트
    """
    # Set image file name.
    img_name = filename + '.png'

    # Return input image size.
    img_arr = cv2.imread(img_name)
    height, width, channel = img_arr.shape

    # Set outermost line.
    line1 = cv2.line(img_arr, (0,0), (height,0), color = (0,0,0), thickness = 1)
    line2 = cv2.line(img_arr, (0,0), (0,width), color = (0,0,0), thickness = 1)
    line3 = cv2.line(img_arr, (height,0), (height,width), color = (0,0,0), thickness = 1)
    line4 = cv2.line(img_arr, (0,width), (height,width), color = (0,0,0), thickness = 1)

    # Set offset value.
    offset = int(height* 0.01)

    # Exclude objects that meet outmost line.
    for index, cnt in enumerate(contour_list):
        tm = tuple(cnt[cnt[:,:,1].argmin()][0])
        bm = tuple(cnt[cnt[:,:,1].argmax()][0])
        lm = tuple(cnt[cnt[:,:,0].argmin()][0])
        rm = tuple(cnt[cnt[:,:,0].argmax()][0])

        if tm[1] <=offset or bm[1] >= height - offset or lm[0] <= offset or rm[0] >= width - offset:
            if index in th_index:
                th_index.remove(index)

    return th_index


def Delete_Depth_Error(contour_list, array_3d, th_index):
    """ 깊이 이상치를 갖는 개체 배제해주는 함수.  
    Args:
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
        th_index: 경계에서 잘린 개체를 제외한 나머지 개체의 인덱스가 저장된 리스트

    Returns:
        th_index: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트
    """

    # Calculate centroid points of all objects(+butterfly shape and broken shape exceptions)
    ctr_list = []
    pop_list = []
    for i in range(len(contour_list)):
        M= cv2.moments(contour_list[i])
        if M["m00"] != 0:
            ctr_list.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])

        elif M["m00"] == 0:
            print(f"{i}번째 인스턴스에서 비정상 contour가 발생하였습니다")
            ctr_list.append([0, 0])
            pop_list.append(i)

        else: 
            pass
    ctr_list = np.array(ctr_list) 
    
    # Apply exception handling.
    pop_list = list(set(pop_list)) # Prevent deduplication
    for k in pop_list:
        if k in th_index:
            th_index.remove(k)
        else: 
            pass

    # Calculate centroid points z value of all objects.
    z_c_list = array_3d[ctr_list[:,1], ctr_list[:,0]][:,2]

    # Exclude object that have depth error in centroid point.
    z_c_list = list(map(int,z_c_list))
    mean = sum(z_c_list)/len(z_c_list)

    for i in th_index:
        if z_c_list[i] > mean*1.2 or z_c_list[i] < mean*0.8:
            if i in th_index:
                th_index.remove(i)
                print(f"-----{i}번 개체가 배제되었습니다. -----")
            else:
                pass
        
        else:
            pass

    return th_index