'''
BRODY v0.1 - Main module

'''

from BRODY.Segmentation import *
from BRODY.Convert import *
from BRODY.Exclude_outlying import *

class ChickenWeightPredictor:
    def __init__(self):
        self.threshold_value = 0.7
        self.origin_img_path = '/scratch/dohyeon/BRODY/src/input/rgb' # 입력 RGB file이 저장된 경로
        self.origin_depthmap_path = '/scratch/dohyeon/BRODY/src/input/depth' # 입력 Depthmap file이 저장된 경로
        self.segmented_img_path = '/scratch/dohyeon/BRODY/src/output/save_point1/' # 출력결과를 저장할 경로
        self.arrival_date = [2022, 4, 26, 00]

    def run(self):
        rgb_file_list = natsorted(os.listdir(self.origin_img_path))
        depthmap_file_list = natsorted(os.listdir(self.origin_depthmap_path))

        for i in range(len(rgb_file_list)):
            rgb_file_name = rgb_file_list[i]
            img_name = self.origin_img_path + "/" + rgb_file_list[i]
            depthmap_name = self.origin_depthmap_path + "/" + depthmap_file_list[i]

            # Image Segmentation + contour + centroid point 생성.
            results, threshold_index = Segment_chicken(img_name, self.threshold_value)

            # Instance Segmentation 결과 얻어진 mask에서 contour 생성.
            contour_list, centroid_list, extream_point_list, mask_list, filtered_index = Get_contour(results, threshold_index)

            # contour에서 직선 찾기.
            Find_straight_line(contour_list)
            
            # 현재 일령계산.
            days = Calculate_day(rgb_file_name, self.arrival_date)

            # 2D 픽셀좌표 > 3D 월드좌표 변환.
            contour_list_3d, z_c_list, array_3d = Convert_2D_to_3D_area(img_name, depthmap_name, contour_list, centroid_list)

            # Major, Minor axis 길이 계산.
            major_axis_list, minor_axis_list = Calculate_major_minor_axis(extream_point_list, array_3d)

            # 이미지 경계에서 잘린 개체 배제.
            exclude_boundary_index_list, exclude_boundary_z_c_list = Exclude_boundary_instance(img_name, extream_point_list, filtered_index, z_c_list)

            # 깊이 이상치갖는 개체 배제.
            exclude_depth_err_index_list, exclude_depth_err_mask_list = Exclude_depth_error(exclude_boundary_index_list, exclude_boundary_z_c_list, mask_list) 

            # # 실제면적 계산.  
            area_list, average_area, perimeter_list, average_perimeter = Calculate_2d_area(contour_list_3d, exclude_depth_err_index_list)

            # perpendicular_point = Calculate_perpendicular_point(array_3d)
            # distance_list = Calculate_distance(array_3d, centroid_list, perpendicular_point, exclude_depth_err_index_list) # 수선의발에서 물체까지

            # # 평균체중 예측.
            # predict_weight_list, predict_average_weight, num_chicken = Calculate_weight(exclude_depth_err_index_list, days, area_list, perimeter_list, major_axis_list)
           
            # # 체중 시각화.
            # Visualize_weight(
            #     origin_img_path, segmented_img_path, results,
            #     predict_weight_list,date, exclude_depth_err_index_list,
            #     predict_average_weight, days, num_chicken,real_z_c_list,
            #     average_area, threshold_value, area_list, rgb_file_name)

            # # # # # csv 파일에 저장.
            # Save_to_csv1(days, predict_average_weight)
            # Save_to_csv2(days, predict_weight_list, keys)

            #추론이 끝나면 입력 및 중간 디렉토리 비우기.
            # Empty_dir('/scratch/dohyeon/mmdetection/input/RGB')
            # Empty_dir('/scratch/dohyeon/mmdetection/input/DEPTH')
            # Empty_dir('/scratch/dohyeon/mmdetection/output/save_point1')

            
if __name__ == "__main__":
    t = ChickenWeightPredictor()
    t.run()