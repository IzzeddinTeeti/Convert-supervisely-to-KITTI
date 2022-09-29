# import the needed modules
import open3d as o3d
import numpy as np
import supervisely as sly
from supervisely.io.fs import remove_dir
from open3d._ml3d.datasets.utils import BEVBox3D
import os
import argparse
import random


def pcd2bin(pcd_file: str, bin_file: str) -> None:
    """
    Convert pcd to bin.

    Args:
        - pcd_file: path to the pcd file.
        - bin_file: path to the bin file.

    Returns:
        None.
    """

    pcloud = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcloud.points, dtype=np.float32)
    intensity = np.asarray(pcloud.colors, dtype=np.float32)[:, 0:1]
    if len(intensity) == 0:
        intensity = np.ones((points.shape[0], 1))
    points = np.hstack((points, intensity)).flatten().astype("float32")
    points.tofile(bin_file)


def to_xyzwhlr(obj: BEVBox3D) -> np.ndarray:
    """Returns box in the common 7-sized vector representation: (x, y, z, w,
    l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
    the width, length and height of the box a is the yaw angle.

    Args:
        obj: BEVBox3D object.

    Returns:
        box: numpy array (7,)
    """
    bbox = np.zeros((7,))
    bbox[0:3] = obj.center - [0, 0, obj.size[1] / 2] # obj.center
    bbox[3:6] = np.array(obj.size)[[0, 2, 1]] #  obj.size
    bbox[6] = obj.yaw

    return bbox


def to_kitti_format(obj: BEVBox3D) -> str:
    """
    Generate KITTI format annotation string from a BEVBox3D object.

    Args:
        obj: BEVBox3D object.

    Returns:
        kitti_str: KITTI format annotation string.
    """

    label_class = "human"  # Describes the type of object
    truncation = -1 # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries 
    occlusion = -1 # Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
    box2d = [0, 0, 0, 0] # 2D bounding box of object in the image
    score = 1 # Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.

    # For the self custom code
    # box = to_xyzwhlr(obj)
    # center = box[:3]
    # size = box[3:6]
    # ry = box[6]
    
    # x, z = center[0], center[2]
    # beta = np.arctan2(z, x)
    # alpha = -np.sign(beta) * np.pi / 2 + beta + ry

    # For the cutom code provided by OpenPCD
    center = obj.center
    size = obj.size
    ry = obj.yaw

    
    # For the self custom code
    # kitti_str = (
    #     "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"
    #     % (
    #         label_class,
    #         truncation,
    #         occlusion,
    #         alpha,
    #         box2d[0],
    #         box2d[1],
    #         box2d[2],
    #         box2d[3],
    #         size[0],
    #         size[1],
    #         size[2],
    #         center[0],
    #         center[1],
    #         center[2],
    #         ry,
    #         score,
    #     )
    # )

    # For the cutom code provided by OpenPCD
    kitti_str = (
        "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %s"
        % (
            center[0],
            center[1],
            center[2],
            size[0],
            size[1],
            size[2],
            ry,
            label_class
        )
    )
    # print(kitti_str)
    return kitti_str


def json2txt(annotation_path: str, txt_path: str, meta: str) -> None:
    """
    Convert json Supervisely annotation to txt KITTI annotation.

    Args:
        annotation_path: path to the json annotation file.
        txt_path: path to the txt annotation file.
        meta: meta data of the dataset.

    Returns:
        None.
    """
    ann_json = sly.json.load_json_file(annotation_path)
    ann = sly.PointcloudAnnotation.from_json(ann_json, meta)
    objects = []

    for fig in ann.figures:
        geometry = fig.geometry
        class_name = fig.parent_object.obj_class.name

        dimensions = geometry.dimensions
        position = geometry.position
        rotation = geometry.rotation
        
        obj = BEVBox3D(
            center=np.array([float(position.x), float(position.y), float(position.z)]),
            size=np.array([float(dimensions.x), float(dimensions.y), float(dimensions.z)]),
            yaw=np.array(float(-rotation.y)),
            label_class=class_name,
            confidence=1.0,
        )  # , world_cam=calib['world_cam'], cam_img=calib['cam_img']
        # print("center", np.array([float(position.x), float(position.y), float(position.z)]), "\n", "size", np.array([float(dimensions.x), float(dimensions.y), float(dimensions.z)]))
        objects.append(obj)

    with open(txt_path, "w") as f:
        for box in objects:
            f.write(to_kitti_format(box))
            f.write("\n")


def main():
    """
    The main function. It converts pointclouds from .pcd to .bin and convert annotations from .json Supervisely to .txt KITTI format.

    Args:
        -input_path: path to the pointclouds
        -output_path: path to the bin files

    Returns:
        None
    """
    ## Add parser
    parser = argparse.ArgumentParser(
        description="Convert pointclouds from .pcd to .bin and the annotation from .json to .txt KITTI format"
    )
    parser.add_argument(
        "--input_path",
        help="Supervisely folder path that contains pointcloud folder, json folder, and meta file",
        type=str,
        default="/mnt/mars-beta/izzeddin/MAESTRO/2_20220525_PILOT", #   /mnt/mars-beta/izzeddin/MAESTRO/testing ann
    )
    parser.add_argument(
        "--output_path",
        help="KITTI folder path that will contain bin folder and txt folder",
        type=str,
        default="/mnt/mars-beta/izzeddin/OpenPCDet/data/custom", #  /mnt/mars-beta/izzeddin/MAESTRO/testing ann
    )
    
    args = parser.parse_args()

    # Percentage of data for training 
    train_percentage = 1 # 80% training, 20% validation

    # Origional folder number to be added to the output files names
    folder_num = "2_"

    # locate the path of four folders and 1 meta file.
    pcd_path = os.path.join(args.input_path, "pointcloud")
    json_path = os.path.join(args.input_path, "ann")
    meta_path = os.path.join(args.input_path, "meta.json")

    # Load meta file
    meta_json = sly.json.load_json_file(meta_path)
    meta = sly.ProjectMeta.from_json(meta_json)

    # Initiate a counter and create a list of all the pcd files in the directory
    counter = 1
    
    pcd_list_full=os.listdir(pcd_path)
    pcd_list=[x.split('.')[0] for x in pcd_list_full]

    json_list_full=os.listdir(pcd_path)
    json_list=[x.split('.')[0] for x in json_list_full]

    training_num = int(len(pcd_list) * train_percentage)
    training_list = random.sample(pcd_list, training_num)
    testing_list = list(set(pcd_list) - set(training_list))
    

    for split_name, split_list in zip(["training", "testing"], [training_list, testing_list]):
        # create the txt file in ImageSets folder
        lists_path = os.path.join(args.output_path, "ImageSets")
        os.makedirs(lists_path, exist_ok=True)

        str_split_list = [folder_num + str(x) for x in split_list]
        if split_name == "training":
            file = os.path.join(lists_path, "train.txt")
            with open(file, "w") as f:
                f.write("\n".join(str_split_list))

        elif split_name == "testing":
            file = os.path.join(lists_path, "val.txt")
            with open(file, "w") as f:
                f.write("\n".join(str_split_list))
            
            file = os.path.join(lists_path, "test.txt")
            with open(file, "w") as f:
                f.write("\n".join(str_split_list))
            

        for pcd in split_list:
            if pcd not in json_list:
                continue
            
            # Convert pcd to bin
            bin_name = folder_num + pcd + ".bin"
            bin_path = os.path.join(args.output_path, split_name, "pointcloud")
            os.makedirs(bin_path, exist_ok=True)
            bin_file = os.path.join(bin_path, bin_name)
            pcd_file = os.path.join(pcd_path, pcd + ".pcd")
            pcd2bin(pcd_file, bin_file)

            # Convert json to txt
            txt_name = folder_num + pcd + ".txt"
            txt_path = os.path.join(args.output_path, split_name, "ann")
            os.makedirs(txt_path, exist_ok=True)
            txt_file = os.path.join(txt_path, txt_name)
            json_file = os.path.join(json_path, pcd + ".pcd.json")
            json2txt(json_file, txt_file, meta)

            # Print the progress
            print("Progress: %d/%d" % (counter, len(pcd_list)))
            counter += 1


if __name__ == "__main__":
    main()

# python test.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 2 --ckpt /mnt/mars-beta/izzeddin/OpenPCDet/output/custom_models/pv_rcnn/default/ckpt/checkpoint_epoch_14.pth
# python train.py --cfg_file cfgs/custom_models/pv_rcnn.yaml --batch_size 2 --workers 1