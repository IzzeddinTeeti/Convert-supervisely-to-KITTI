# Convert supervisely to KITTI
Convert LiDAR pointclouds and their annotations from Supervisely to KITTI format.

It converts pointclouds from .pcd format to .bin format and 
converts annotations from .json format to .txt format.

## Run the code
1. Install the requirements from `requirements.txt` using `pip install -r requirements.txt`.
2. Run the conversion code using 
```
python convert_main.py --pcd_path=path/to/pcd/folder 
                       --bin_path=path/to/save/bin/files 
                       --meta_path=path/to/meta.json/produced/by/supervisly 
                       --json_path=path/to/json/folder 
                       --txt_path=path/to/save/txt/annotations/files
```
