
from data_handler import ImageProcessing
processed_folder = r"D:\DATA\Lab rotation\PROCESSED"
raw_folder = [r"D:\DATA\Lab rotation\RAW\2024-11-27_ST_D_PNG",r"D:\DATA\Lab rotation\RAW\2024-11-28_ST_D_PNG"]

for folder in raw_folder:
    ImageProcessing.preprocess_dataset(folder, processed_folder, 1)

# from file_handler import FileReader
# processed_file = r"D:\DATA\Lab rotation\PROCESSED\2024-11-27_ST_D_PNG.h5"
# datasets,attributes = FileReader.load_HDF5(processed_file,'coordinates')

# points = datasets['point_cloud/xyz_coordinates']

# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])