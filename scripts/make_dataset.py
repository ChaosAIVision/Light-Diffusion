
# from datasets import load_dataset
# dataset = load_dataset("ArtMancer/ORiDa-50-samples", split="train")
# dataset_save_path = "/home/chaos/Documents/chaos/repo/Light-Diffusion/dataset/object_insertion"
# # for i in range(len(dataset)):
# #     target_image = dataset[i]['image']
# #     object_image = dataset[i]['ground_truth']
# #     mask = dataset[i]['masks']
# #     target_image.save(f"{dataset_save_path}/target_image_{i}.png")
# #     object_image.save(f"{dataset_save_path}/object_image_{i}.png")
# #     mask.save(f"{dataset_save_path}/mask_{i}.png")

import csv
from pathlib import Path

def generate_dataset_csv(dataset_dir: str, output_csv: str) -> None:
    """
    Generate a CSV mapping target_image, object_image, and mask file paths.

    Args:
        dataset_dir (str): Directory containing saved images.
        output_csv (str): Output CSV file path.
    """
    dataset_path = Path(dataset_dir)
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all target images
    target_images = sorted(dataset_path.glob("target_image_*.png"))
    object_images = sorted(dataset_path.glob("object_image_*.png"))
    masks = sorted(dataset_path.glob("mask_*.png"))

    if not (len(target_images) == len(object_images) == len(masks)):
        raise ValueError("Image counts are not equal across target/object/mask sets.")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_image", "object_image", "mask"])

        for t_img, o_img, m_img in zip(target_images, object_images, masks):
            writer.writerow([t_img.resolve(), o_img.resolve(), m_img.resolve()])


if __name__ == "__main__":
    DATASET_DIR = "/home/chaos/Documents/chaos/repo/Light-Diffusion/dataset/object_insertion"
    OUTPUT_CSV = "/home/chaos/Documents/chaos/repo/Light-Diffusion/dataset/object_insertion/dataset_paths.csv"

    generate_dataset_csv(DATASET_DIR, OUTPUT_CSV)
