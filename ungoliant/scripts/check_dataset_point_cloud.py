from data_handling.file_handler import FileReader 
from data_visualization.data_visualizer import Open3D

INPUT_FILEPATH = r"D:\DATA\Lab rotation\PROCESSED\2024-11-28_ST_D_PNG.h5"
datasets, attributes = FileReader.load_HDF5(INPUT_FILEPATH, "coordinates")
points = datasets["point_cloud/xyz_coordinates"]
Open3D.plot_point_cloud(points)
