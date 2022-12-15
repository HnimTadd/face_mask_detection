import os


data_dir = "./data"
if not os.path.exists(data_dir):
    print("Create data folder")
    os.mkdir(data_dir)