import os
from dataclasses import dataclass
from os import listdir, path, makedirs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import imageio
import tensorflow_datasets as tfds
from tqdm.auto import trange


@dataclass
class Args:
    save_root: str = "path_to/openx"  # Root for the videos to save
    orig_root: str = "path_to/rtx"  # Root for the downloaded videos


def dataset2path(dataset_name) -> str:
    versions = listdir(path.join(Args.orig_root, dataset_name))
    versions.sort()
    versions = [version for version in versions if len(version) == 5]
    version = versions[-1]
    return path.join(Args.orig_root, dataset_name, version)


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def extract_sample(tfds_builder, obs_key: str, dataset_name: str, save_dir: str, split: str, extra: str = None) -> None:
    try:
        ds = tfds_builder.as_dataset(split=split)
    except:
        ds = tfds_builder.as_data_source(split=split)
    ds_iter = iter(ds)
    for episode_idx in trange(len(ds), desc=f"Extracting {dataset_name.upper()} {split}"):
        try:
            episode = next(ds_iter)
            if dataset_name == "robot_vqa":
                images = []
                for step in episode["steps"]:
                    images.extend([img for img in step["observation"][obs_key]])
            elif extra is None:
                images = [step["observation"][obs_key] for step in episode["steps"]]
            else:
                images = [step["observation"][obs_key][extra] for step in episode["steps"]]
            try:
                images = [image.numpy() for image in images]
            except:
                images = images

            save_path = path.join(save_dir, split, f"{episode_idx:08}.mp4")
            makedirs(path.dirname(save_path), exist_ok=True)

            save_images_to_video(images, save_path)
        except:
            pass


dataset_list = [
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
]
feasible_datasets = 0
infeasible_datasets = []
display_keys = [
    "image", "wrist_image", "hand_image", "top_image", "wrist225_image", "wrist45_image", "image_manipulation",
    "highres_image", "finger_vision_1", "finger_vision_2", "image_fisheye", "wrist_image_left",
    "image_side_1", "image_side_2", "image_wrist_1", "image_wrist_2", "image_additional_view",
    "image_left_side", "image_right_side", "image_left", "image_right", "image_top", "image_wrist",
    "front_image_1", "front_image_2", "exterior_image_1_left", "exterior_image_2_left",
    "frontleft_fisheye_image", "frontright_fisheye_image", "hand_color_image",
    "rgb", "front_rgb", "agentview_rgb", "eye_in_hand_rgb", "rgb_static", "rgb_gripper",
    "image_1", "image_2", "image_3", "image_4", "image1", "image2", "images",
    "cam_high", "cam_left_wrist", "cam_right_wrist"
]
for dataset in dataset_list:
    is_feasible = False
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
    for display_key in display_keys:
        if display_key in builder.info.features["steps"]["observation"]:
            if dataset == "mimic_play":
                if display_key == "image":
                    folder = path.join(Args.save_root, f"{dataset}-front_image_1")
                    if not path.exists(folder):
                        for split_name in builder.info.splits.keys():
                            extract_sample(builder, display_key, dataset, folder, split_name, "front_image_1")
                    folder = path.join(Args.save_root, f"{dataset}-front_image_2")
                    if not path.exists(folder):
                        for split_name in builder.info.splits.keys():
                            extract_sample(builder, display_key, dataset, folder, split_name, "front_image_2")
                else:
                    folder = path.join(Args.save_root, f"{dataset}-{display_key}")
                    if not path.exists(folder):
                        for split_name in builder.info.splits.keys():
                            extract_sample(builder, display_key, dataset, folder, split_name, display_key)
            else:
                folder = path.join(Args.save_root, f"{dataset}-{display_key}")
                if not path.exists(folder):
                    for split_name in builder.info.splits.keys():
                        extract_sample(builder, display_key, dataset, folder, split_name)
            is_feasible = True
    if is_feasible:
        feasible_datasets += 1
    else:
        infeasible_datasets.append(dataset)
print("Feasible datasets:", feasible_datasets)
print("Infeasible datasets:", infeasible_datasets)
