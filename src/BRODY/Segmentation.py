'''
BRODY v0.1 - Segmentation module

'''

from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def Segment_chicken(img_name):
    """입력 이미지(육계군집 영상)에 Instance Segmentation을 적용하여 육계영역만 분할해주는 함수.

    Args:
        img_name: 입력 이미지를 불러오는 디렉토리 경로.

    Returns:
        results: Detector 추론 결과 얻어진 bbox, mask 정보
        filtered_index: confidence score가 threshold값 보다 큰 인스턴스의 index 정보
    """

    # config 파일을 설정하고, 학습한 checkpoint file 불러오기.
    # config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_15.py' # Mask-RCNN-Dataset_15
    # checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_15/epoch_36.pth' # Mask-RCNN-Dataset_15
    config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_30.py' # Mask-RCNN-Dataset_30
    checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_30/epoch_36.pth' # Mask-RCNN-Dataset_30
    # config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_68.py' # Mask-RCNN-Dataset_68
    # checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_68/epoch_35.pth' # Mask-RCNN-Dataset_68
    # config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_87.py' # Mask-RCNN-Dataset_87
    # checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_87/epoch_35.pth' # Mask-RCNN-Dataset_87

    # config 파일과 checkpoint를 기반으로 Detector 모델을 생성.
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # rgb image 불러오기. 
    img_arr= cv2.imread(img_name, cv2.IMREAD_COLOR)

    # inference 진행 및 결과 array 형태로 저장.
    results = inference_detector(model, img_arr)

    # confidence score가 threshold값 보다 큰 인스턴스의 index 정보 저장.
    threshold_index = np.where(results[0][0][:,4]> 0.7)
    threshold_index = threshold_index[0].tolist()

    return results, threshold_index

def Get_contour(results, threshold_index):
    """Instance Segmentation 결과 얻어진 mask에서 contour 생성해주는 함수.

    Args:
        results: Segmentation 결과 얻은 bbox, mask 정보
        threshold_index: confidence score가 threshold값 보다 큰 인스턴스의 index 정보

    Returns:
        contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
        centroid_list (ndarray): 모든 개체의 무게중심점 픽셀좌표가 저장된 리스트
        extream_point_list: 모든 개체의 상하좌우 극점이 저장된 리스트
        mask_list: instace를 이루는 모든 픽셀좌표가 저장된 리스트
        filtered_index: 예외처리된 모든 개체의 인덱스가 저장된 리스트
    """
    
    # 개별 인스턴스의 contour 픽셀좌표가 저장될 list선언.
    contour_list = []
    centroid_list = []
    extream_point_list = []
    mask_list = []
    pop_list = []

    for i in threshold_index:
        # 개별 인스턴스의 binary mask를 array 형태로 받아오기.
        mask_array = np.where(results[1][0][i]==1, 255, results[1][0][i]).astype(np.uint8)
        pixels = cv2.findNonZero(mask_array)
        mask_list.append(pixels)

        # contour 생성.
        contour, hierarchy = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list.append(contour) # contour_list에 contour 좌표 append
        cnt = contour[0]
        
        # contour의 극점 추출(상하좌우 극점)
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        extream_point_list.append([topmost, bottommost, leftmost, rightmost])

        # Contour 예외처리1(하나의 인스턴스에 2개 이상의 contour가 있는 경우)
        if len(contour) == 1:
            pass
        else:
            pop_list.append(i)
            print(f"{i}번째 인스턴스는 contour가 2개 그려집니다.")

        # Contour 예외처리2(contour를 이루는 point 개수가 1~3개인 경우)
        if len(contour[0]) in [1,3]:
            pop_list.append(i)
            print(f"{i}번째 인스턴스에서 contour point가 1~3개입니다")
        else: 
            pass

    # Contour 예외처리3(나비모양, 끊긴모양)
    for i in threshold_index:
        M= cv2.moments(contour_list[i][0])
        if M["m00"] != 0: # 정상
            centroid_list.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])

        elif M["m00"] == 0: # 비정상(나비모양, 끊긴모양)
            print(f"{i}번째 인스턴스에서 비정상 contour가 발생하였습니다")
            centroid_list.append([0, 0])
            pop_list.append(i)

        else: 
            pass

    # 예외처리 적용(하나에 인스턴스에 2개 이상의 contour 형성된 경우, contour를 이루는 point 개수가 1~3개인 경우, 나비모양 및 끊긴모양)
    pop_list = list(set(pop_list)) # 중복제거 방지
    for k in pop_list:
        threshold_index.remove(k)

    filtered_index = threshold_index
    contour_list = np.array(contour_list)
    centroid_list = np.array(centroid_list)

    return contour_list, centroid_list, extream_point_list, mask_list, filtered_index