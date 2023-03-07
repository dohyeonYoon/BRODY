'''
BRODY v0.1 - Visualization module

'''

# from mmdet.apis import init_detector
from csv import writer
from datetime import datetime
from natsort import natsorted
from mmdet.apis import init_detector
import os 
import cv2

def Calculate_day(rgb_file_name, arrival_date):
    """촬영당시 육계의 일령 계산해주는 함수.

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


def Visualize_weight(input_path,
                    output_path,
                    results, 
                    weight_list, 
                    exclude_depth_err_index_list, 
                    average_weight, 
                    days,
                    average_area, 
                    area_list,
                    rgb_file_name,):
    """이미지 내의 각 개체마다 id, 체중 시각화해주는 함수.

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
    # config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_15.py' # Mask-RCNN-Dataset_15
    # checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_15/epoch_36.pth' # Mask-RCNN-Dataset_15
    # config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_30.py' # Mask-RCNN-Dataset_30
    # checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_30/epoch_36.pth' # Mask-RCNN-Dataset_30
    # config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_68.py' # Mask-RCNN-Dataset_68
    # checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_68/epoch_35.pth' # Mask-RCNN-Dataset_68
    config_file = '/scratch/dohyeon/BRODY/src/method_override/config/num_dataset_87.py' # Mask-RCNN-Dataset_87
    checkpoint_file = '/scratch/dohyeon/BRODY/src/method_override/weights/mask_rcnn_r101/num_dataset_87/epoch_35.pth' # Mask-RCNN-Dataset_87

    # config 파일과 checkpoint를 기반으로 Detector 모델을 생성.
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    # 입력 이미지에 추론결과 visualize
    for i in range(1):
        img_name = input_path + '/' + rgb_file_name
        img_arr= cv2.imread(img_name, cv2.IMREAD_COLOR)

        # 추론결과 디렉토리에 저장(confidenece score 0.7이상의 instance만 이미지에 그릴 것).
        model.show_result(img_arr,
                        results,
                        area_list,
                        weight_list,
                        exclude_depth_err_index_list,
                        days,
                        average_area,
                        average_weight,
                        score_thr=0.7,
                        bbox_color=(0,0,0),
                        text_color=(255, 255, 255),
                        thickness=0.01,
                        font_size=12,
                        out_file= f'{output_path}{rgb_file_name}')

    return


def Save_to_csv(days, predict_average_weight):
    """각 개체의 일령, 면적, 체중을 csv file에 저장해주는 함수.

    Args:
        date: 영상이 촬영된 날짜.
        days: 일령.
        area_list: 각 instance의 면적이 저장된 list.
        predict_weight_list: 각 instance의 예측체중이 저장된 list.

    Returns:
        None
    """
    # 각 개체의 일령, 면적, 체중 3가지를 csv file에 저장.
    rows = [days, predict_average_weight]
    with open(f'/scratch/dohyeon/mmdetection/output/avg_weight_result.csv','a', newline='') as f_object:
        # using csv.writer method from CSV package
        writer_object = writer(f_object)
        writer_object.writerow(rows)
        f_object.close()

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