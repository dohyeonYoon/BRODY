# '''
# BRODY v0.1 - Prediction module

# '''

# from shapely.geometry import Polygon
# import numpy as np 
# import joblib

# def Calculate_2D_Area(boundary_point_list, th_index):
#     """3차원 contour점으로 이루어진 다각형의 면적 계산해주는 함수.

#     Args:
#         contour_list (ndarray): 모든 개체의 contour점 픽셀좌표가 저장된 리스트
#         array_3d: 모든 픽셀의 3차원 좌표가 저장된 array
#         th_index: 깊이 이상치 개체를 제외한 나머지 개체 인덱스가 저장된 리스트

#     Returns:
#         area_list: 모든 개체의 면적이 저장된 리스트
#     """

#     # Calculate area of all objects.
#     area_list = []
#     for i in th_index:
#         polygon = Polygon(np.array(boundary_point_list[i]))
#         polygon_area = round((polygon.area)/100,2)
#         area_list.append(polygon_area)

#     return area_list


# def Calculate_Weight(area_list):
#     """회귀방정식을 통해 면적으로부터 체중을 예측해주는 함수.

#     Args:
#         area_list: 모든 개체의 면적이 저장된 리스트

#     Returns:
#         weight_list: 모든 개체의 예측체중이 저장된 리스트
#     """
#     # Load linear regression model.
#     model = joblib.load('./regression/weights/optimal.pkl')

#     # Calculate weight of all objects.
#     weight_list = []
#     for i in range(len(area_list)):
#         area = area_list[i]

#         weight = model.predict([[area]])
#         weight_list.append(weight[0][0])

#     return weight_list

'''
BRODY v0.1 - Prediction module

'''

from shapely.geometry import Polygon
import numpy as np 
import joblib
import open3d as o3d
import cv2


def Calculate_2D_Area(contour_list, array_3d, th_index):
    """3차원 contour점으로 이루어진 다각형의 면적,둘레 계산해주는 함수.

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
    perimeter_list = []
    for i in th_index:
        polygon = Polygon(np.array(contour_list_3d[i]))
        polygon_area = round((polygon.area)/100,3)
        polygon_perimeter = round(polygon.length, 3)
        area_list.append(polygon_area)
        perimeter_list.append(polygon_perimeter)

    return area_list, perimeter_list


def Calculate_alpha(mask_list_3d, th_index):
    '''
    카메라(O)와 물체(X)가 이루는 사이각 계산
                          O
                        / |
                       /  |
                      /   |
                     /    |
                    /     |
                   X 
    '''
    # Remove outlier points
    filtered_mask_list_3d = []
    for i in th_index:
        mask_point = o3d.geometry.PointCloud()
        mask_point.points = o3d.utility.Vector3dVector(mask_list_3d[i])
        mask_point.normals = o3d.utility.Vector3dVector(mask_list_3d[i])

        filtered_mask_point,ind = mask_point.remove_statistical_outlier(nb_neighbors=60, std_ratio=0.8)
        filtered_mask_point = np.asarray(filtered_mask_point.points)
        filtered_mask_list_3d.append(filtered_mask_point)

    center_of_gravity_list = []
    for i in filtered_mask_list_3d:
        center_x = sum(i[:,0]) / len(i[:,0])
        center_y = sum(i[:,1]) / len(i[:,1])
        center_z = sum(i[:,2]) / len(i[:,2])
        center_of_gravity_list.append([center_x, center_y, center_z])
    
    unit_vector_instance_list = []
    for i in center_of_gravity_list:
        unit_vector_instance_list.append(i / np.linalg.norm(i))
    unit_vector_instance_list = np.array(unit_vector_instance_list)

    unit_vector_vertical = np.array([0.0,0.0, 1.0])

    # 두 단위 벡터 사이의 각도 반환
    alpha_list = []
    for i in unit_vector_instance_list:
        angle_radian = np.arccos(np.clip(np.dot(unit_vector_vertical, i), -1.0, 1.0))
        angle_degree = angle_radian*(180/np.pi)
        alpha_list.append(angle_degree)
    alpha_list = np.array(alpha_list)

    return alpha_list


def Calculate_beta(filename, contour_list, th_index):
    '''
    물체의 orientation 방향, 장축, 단축 교차점 계산해주는 함수
    '''

    # Return input image size.
    img_name = filename + '.png'
    height, width, channel = cv2.imread(img_name).shape

    beta_list = []
    intersect_point_list = []
    perpendicular_point_list = []
    for i in th_index:
        # Generate empty mask and draw only contour points
        mask = np.zeros((height, width))
        mask = cv2.drawContours(mask, [contour_list[i]], -1, 255, 2)
        
        # Calculate centroid point.
        M = cv2.moments(contour_list[i])
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Reset the longest line.
        max_distance = 0
        max_angle = 0
        longest_line = None

        # Generate line with 1-degree intervals.
        for angle in np.linspace(0, 2 * np.pi, 360):
            # Calculate direction vector.
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Draw two lines: original direction, opposite direction.
            end_point1 = (int(cx + dx * 10000), int(cy + dy * 10000))
            end_point2 = (int(cx - dx * 10000), int(cy - dy * 10000))

            line1 = cv2.line(mask.copy(), (cx, cy), end_point1, 127, 1)
            line2 = cv2.line(mask.copy(), (cx, cy), end_point2, 127, 1)

            # Find the points where the two lines intersect with the contour.
            y1, x1 = np.where((line1 == 127) & (mask == 255))
            y2, x2 = np.where((line2 == 127) & (mask == 255))

            # In the case that intersection points occur in both directions.
            if len(x1) >= 1 and len(x2) >= 1:
                distance = np.sqrt((x1[-1] - x2[0]) ** 2 + (y1[-1] - y2[0]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    max_angle = angle
                    longest_line = [[x2[0], y2[0]], [x1[-1], y1[-1]]]

        max_angle = np.degrees(max_angle)
        perpendicular_angle = max_angle + 90
        perpendicular_angle = np.radians(perpendicular_angle)

        # Calculate perpendicular direction vector.
        dx_p = np.cos(perpendicular_angle)
        dy_p = np.sin(perpendicular_angle)

        # Draw two lines: original direction, opposite direction.
        end_point_p1 = (int(cx + dx_p * 2000), int(cy + dy_p * 2000))
        end_point_p2 = (int(cx - dx_p * 2000), int(cy - dy_p * 2000))

        line_p1 = cv2.line(mask.copy(), (cx, cy), end_point_p1, 127, 1)
        line_p2 = cv2.line(mask.copy(), (cx, cy), end_point_p2, 127, 1)

        # Find the points where the two lines intersect with the contour.
        y1_p, x1_p = np.where((line_p1 == 127) & (mask == 255))
        y2_p, x2_p = np.where((line_p2 == 127) & (mask == 255))
        perpendicular_line = [[x2_p[0], y2_p[0]], [x1_p[-1], y1_p[-1]]]

        # # 테스트!!!!!!
        # cv2.line(mask, longest_line[0], longest_line[1], 255, 1)
        # cv2.circle(mask, (cx, cy), 1, 255, -1)
        # cv2.circle(mask, (longest_line[0]), 3, 255, -1)
        # cv2.circle(mask, (longest_line[1]), 3, 255, -1)
        # cv2.line(mask, perpendicular_line[0], perpendicular_line[1], 255, 1)
        # cv2.circle(mask, (perpendicular_line[0]), 3, 255, -1)
        # cv2.circle(mask, (perpendicular_line[1]), 3, 255, -1)
        # cv2.imwrite(f"./output/mask/{i}.png",mask)

        beta_list.append(max_angle)
        intersect_point_list.append(longest_line)
        perpendicular_point_list.append(perpendicular_line)

    return beta_list, intersect_point_list, perpendicular_point_list


def Calculate_major_dist(array_3d, intersect_point_list):
    '''
    장축길이 계산하는 함수
    '''
    # Calculate major distance.
    major_dist_list = []
    for i in intersect_point_list:
        x1,y1 = i[0][0], i[0][1]
        x2,y2 = i[1][0], i[1][1]
        intersect_point1_3d = array_3d[y1,x1]
        intersect_point2_3d = array_3d[y2,x2]
        distance = np.sqrt((intersect_point1_3d[0] - intersect_point2_3d[0]) ** 2 + (intersect_point1_3d[1] - intersect_point2_3d[1]) ** 2)
        major_dist_list.append(distance)
    
    return major_dist_list


def Calculate_minor_dist(array_3d, perpendicular_point_list):
    '''
    단축길이 계산하는 함수
    '''
    # Calculate major distance.
    minor_dist_list = []
    for i in perpendicular_point_list:
        x1,y1 = i[0][0], i[0][1]
        x2,y2 = i[1][0], i[1][1]
        perpendicular_point1_3d = array_3d[y1,x1]
        perpendicular_point2_3d = array_3d[y2,x2]
        distance = np.sqrt((perpendicular_point1_3d[0] - perpendicular_point2_3d[0]) ** 2 + (perpendicular_point1_3d[1] - perpendicular_point2_3d[1]) ** 2)
        minor_dist_list.append(distance)

    return minor_dist_list


def Get_surface_area(mask_list_3d, th_index):
    surface_area_list = []
    for i in th_index:
        mask_point = o3d.geometry.PointCloud()
        mask_point.points = o3d.utility.Vector3dVector(mask_list_3d[i])
        mask_point.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        mask_point.paint_uniform_color([0,0.5,0])

        # Calculate normals.
        mask_point.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=150))

        # Correct and reverse normals.
        mask_point.orient_normals_consistent_tangent_plane(100)
        normals = np.asarray(mask_point.normals) # Reverse normal vector
        mask_point.normals = o3d.utility.Vector3dVector(-normals) # Reverse normal vector
        o3d.visualization.draw_geometries([mask_point], point_show_normal=True)

        # Ball pivoting algorithm
        radii = [8, 2, 4, 8]
        ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(mask_point, o3d.utility.DoubleVector(radii)) # Creating a mesh using the Ball-Pivoting algorithm
        ball_surface_area = ball_mesh.get_surface_area() # Compute surface area of the mesh
        print(f"Surface area for index {i}: {ball_surface_area}")

        # Poisson surface algorithm
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mask_point, depth=10)
        vertices_to_remove = densities < np.quantile(densities, 0.25)
        poisson_mesh.remove_vertices_by_mask(vertices_to_remove)
        # bbox = mask_point.get_axis_aligned_bounding_box()
        # poisson_mesh = poisson_mesh.crop(bbox)
        poisson_surface_area = poisson_mesh.get_surface_area() # Compute surface area of the mesh

        print(f"Surface area for index {i}: {poisson_surface_area}")

        # For visualization
        o3d.visualization.draw_geometries([ball_mesh])
        o3d.visualization.draw_geometries([poisson_mesh])

    return surface_area_list


def Calculate_Weight(area_list):

    # Load linear regression model.
    model = joblib.load('./regression/weights/Huber.pkl')

    # Calculate weight of all objects.
    weight_list = []
    for i in range(len(area_list)):
        area = area_list[i]
        weight = model.predict([[area]])
        weight_list.append(weight[0])

    return weight_list

# def Calculate_Weight(surface_area_list):

#     # Load linear regression model.
#     model = joblib.load('./regression/weights/surface_weight.pkl')

#     # Calculate weight of all objects.
#     weight_list = []
#     for i in range(len(surface_area_list)):
#         surface_area = surface_area_list[i]
#         weight = model.predict([[surface_area]])
#         weight_list.append(weight)

#     return weight_list