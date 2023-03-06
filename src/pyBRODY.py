'''
BRODY v0.1 - Main module

'''

import BRODY.Segmentation as Segmentation
import BRODY.Conversion as Conversion
import BRODY.Exclude_Outlying as eo
import BRODY.Prediction as Prediction
import BRODY.Visualization as Visualization

from natsort import natsorted 
import os

# 경로선언
origin_img_path = '/scratch/dohyeon/BRODY/src/input/rgb' # 입력 RGB file이 저장된 경로
origin_depthmap_path = '/scratch/dohyeon/BRODY/src/input/depth' # 입력 Depthmap file이 저장된 경로
segmented_img_path = '/scratch/dohyeon/BRODY/src/output/save_point1/' # 출력결과를 저장할 경로
arrival_date = [2022, 4, 26, 00]
rgb_file_list = natsorted(os.listdir(origin_img_path))
depthmap_file_list = natsorted(os.listdir(origin_depthmap_path))

def main():
    for i in range(len(rgb_file_list)):
        rgb_file_name = rgb_file_list[i]
        img_name = origin_img_path + "/" + rgb_file_list[i]
        depthmap_name = origin_depthmap_path + "/" + depthmap_file_list[i]

        # Segmentation
        results, threshold_index = Segmentation.Segment_chicken(img_name)
        contour_list, centroid_list, extream_point_list, mask_list, filtered_index = Segmentation.Get_contour(results, threshold_index)

        # Conversion
        z_c_list, array_3d, contour_list_3d = Conversion.Convert_2D_to_3D_area(img_name, depthmap_name, contour_list, centroid_list)
        major_axis_list, minor_axis_list = Conversion.Calculate_major_minor_axis(extream_point_list, array_3d)

        # eo
        exclude_boundary_index_list = eo.Exclude_boundary_instance(img_name, extream_point_list, filtered_index, z_c_list)
        exclude_depth_err_index_list = eo.Exclude_depth_error(exclude_boundary_index_list, z_c_list)
        # eo.Find_straight_line(contour_list)

        # # Prediction  
        # area_list, average_area, perimeter_list, average_perimeter = Prediction.Calculate_2d_area(contour_list_3d, exclude_depth_err_index_list)
        # predict_weight_list, predict_average_weight, num_chicken = Prediction.Calculate_weight(exclude_depth_err_index_list, days, area_list, perimeter_list, major_axis_list)
        
        # # Visualization
        # days = Visualization.Calculate_day(rgb_file_name, arrival_date)
        # Visualization.Visualize_weight(origin_img_path, 
        #         segmented_img_path,
        #         results,
        #         predict_weight_list, 
        #         exclude_depth_err_index_list, 
        #         predict_average_weight, 
        #         days,
        #         num_chicken,
        #         average_area, 
        #         threshold_value,
        #         area_list,
        #         rgb_file_name,)
        # Visualization.Save_to_csv(days, predict_average_weight)
        # Visualization.Empty_dir(origin_img_path)
        # Visualization.Empty_dir(origin_depthmap_path)
            
if __name__ == "__main__":
    main()