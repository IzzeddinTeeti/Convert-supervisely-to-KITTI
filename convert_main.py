# import the needed modules
import open3d as o3d
import numpy as np
import supervisely as sly
from supervisely.io.fs import remove_dir
from open3d._ml3d.datasets.utils import BEVBox3D
import os
import argparse
import itertools


def pcd2bin(pcd_path: str, bin_path: str) -> None:
    """
    Convert pcd to bin.

    Args:
        - pcd_path: path to the pcd file.
        - bin_path: path to the bin file.

    Returns:
        None.
    """

    pcloud = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcloud.points, dtype=np.float32)
    intensity = np.asarray(pcloud.colors, dtype=np.float32)[:, 0:1]
    if len(intensity) == 0:
        intensity = np.ones((points.shape[0], 1))
    points = np.hstack((points, intensity)).flatten().astype("float32")
    points.tofile(bin_path)


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
    bbox[0:3] = obj.center - [0, 0, obj.size[1] / 2]
    bbox[3:6] = np.array(obj.size)[[0, 2, 1]]
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

    box2d = [0, 0, 0, 0]
    truncation = -1
    occlusion = -1
    label_class = "human"  # for human
    # confidence = 1.0
    score = 1

    box = to_xyzwhlr(obj)
    center = box[:3]
    size = box[3:6]
    ry = box[6]

    x, z = center[0], center[2]
    beta = np.arctan2(z, x)
    alpha = -np.sign(beta) * np.pi / 2 + beta + ry

    kitti_str = (
        "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"
        % (
            label_class,
            truncation,
            occlusion,
            alpha,
            box2d[0],
            box2d[1],
            box2d[2],
            box2d[3],
            size[0],
            size[1],
            size[2],
            center[0],
            center[1],
            center[2],
            ry,
            score,
        )
    )

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
            size=np.array([float(dimensions.x), float(dimensions.z), float(dimensions.y)]),
            yaw=np.array(float(-rotation.z)),
            label_class=class_name,
            confidence=1.0,
        )  # , world_cam=calib['world_cam'], cam_img=calib['cam_img']
        
        objects.append(obj)

    with open(txt_path, "w") as f:
        for box in objects:
            f.write(to_kitti_format(box))
            f.write("\n")


def main():
    """
    The main function. It converts pointclouds from .pcd to .bin and convert annotations from .json Supervisely to .txt KITTI format.

    Args:
        -pcd_path: path to the pointclouds
        -bin_path: path to the bin files
        -meta_path: path to the meta.json file
        -json_path: path to the json files
        -txt_path: path to the txt files

    Returns:
        None
    """
    ## Add parser
    parser = argparse.ArgumentParser(
        description="Convert pointclouds from .pcd to .bin and the annotation from .json to .txt KITTI format"
    )
    parser.add_argument(
        "--pcd_path",
        help=".pcd folder path",
        type=str,
        default="/mnt/mars-beta/izzeddin/MAESTRO/2_20220525_PILOT/1/pointcloud",
    )
    parser.add_argument(
        "--bin_path",
        help=".bin folder path.",
        type=str,
        default="/mnt/mars-beta/izzeddin/MAESTRO/2_20220525_PILOT/bin/pointcloud",
    )
    parser.add_argument(
        "--meta_path",
        help=".bin folder path.",
        type=str,
        default="/mnt/mars-beta/izzeddin/MAESTRO/2_20220525_PILOT/meta.json",
    )
    parser.add_argument(
        "--json_path",
        help=".json meta file path",
        type=str,
        default="/mnt/mars-beta/izzeddin/MAESTRO/2_20220525_PILOT/1/ann",
    )
    parser.add_argument(
        "--txt_path",
        help=".txt folder path",
        type=str,
        default="/mnt/mars-beta/izzeddin/MAESTRO/2_20220525_PILOT/bin/ann",
    )
    args = parser.parse_args()

    # Load meta file
    path_to_meta = args.meta_path
    meta_json = sly.json.load_json_file(path_to_meta)
    meta = sly.ProjectMeta.from_json(meta_json)

    # Initiate a counter and create a list of all the pcd files in the directory
    counter = 1
    pcd_files = []
    names = []
    for path, _, files in os.walk(args.pcd_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            name = os.path.splitext(filename)[-2]
            if ext == ".pcd":
                pcd_files.append(path + "/" + filename)
                names.append(name)

    pcd_files.sort()
    names.sort()

    # Convert pcd files to bin and convert json files to txt
    for (pcd, json) in zip(pcd_files, os.listdir(args.json_path)):

        bin_path = args.bin_path + "/" + names[pcd_files.index(pcd)] + ".bin"
        pcd2bin(pcd, bin_path)

        name = json.split(".")[0]
        txt_path = os.path.join(args.txt_path, name + ".txt")
        annotation_path = os.path.join(args.json_path, json)
        json2txt(annotation_path, txt_path, meta)

        print("Converted {} of {}".format(counter, len(pcd_files)))
        counter += 1


if __name__ == "__main__":
    main()
