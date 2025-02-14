from data_handling import ImageProcessing

IMG_CHANNEL = 1
OUTPUT_PATH = r"D:\DATA\Lab rotation\PROCESSED"
INPUT_PATH =  r"D:\DATA\Lab rotation\RAW\2025-02-10_LG_G_G_PNG",
ImageProcessing.preprocess_dataset(INPUT_PATH, OUTPUT_PATH, IMG_CHANNEL)