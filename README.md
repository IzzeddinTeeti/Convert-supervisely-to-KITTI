# Convert supervisely to KITTI
Convert LiDAR pointclouds and their annotations from Supervisely to KITTI format.

It converts pointclouds from `.pcd` format to `.bin` format and 
converts annotations from `.json` format to `.txt` format.

The annottaions format in the `.txt` files will be as the following example: 
```
# format: [x y z dx dy dz heading_angle category_name]
0.24 0.18 0.53 0.45 1.77 0.67 1.51 human
```

## Run the code
1. Install the requirements from `requirements.txt` using `pip install -r requirements.txt`.
2. The folders and files directories should be organised as follows:
```
Supervisly_folder
├── pointcloud
│   ├── 1.pcd
│   ├── 2.pcd
│   ├── ...
├── ann
│   ├── 1.json
│   ├── 2.json
│   ├── ...
├── meta.json

KITTI_folder
├── pointcloud
│   ├── 1.bin
│   ├── 2.bin
│   ├── ...
├── ann
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
```

4. Run the conversion code using 
```
python convert_main.py --input_path=Supervisely folder path that contains pointcloud folder, json folder, and meta file 
                       --output_path=KITTI folder path that will contain bin folder and txt folder
```
