import os
import shutil
import random
import argparse

def split_dataset(base_dir, outdir, train_ratio=0.9):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'masks')

    train_images_dir = os.path.join(base_dir, 'train', 'images')
    train_labels_dir = os.path.join(base_dir, 'train', 'masks')
    test_images_dir  = os.path.join(base_dir, 'test', 'images')
    test_labels_dir  = os.path.join(base_dir, 'test', 'masks')

    # Create new directories if they exist not
    for dir_path in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of filenames, assuming they match in images and labels
    filenames = os.listdir(images_dir)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(images_dir, f))]

    # Shuffle the filenames for random split
    random.shuffle(filenames)

    # Compute train/test split
    train_count = int(len(filenames) * train_ratio)
    train_files = filenames[:train_count]
    test_files = filenames[train_count:]

    # Move files
    for file_list, img_dest, lbl_dest in [
        (train_files, train_images_dir, train_labels_dir),
        (test_files, test_images_dir, test_labels_dir)
    ]:
        for fname in file_list:
            img_src = os.path.join(images_dir, fname)
            lbl_src = os.path.join(labels_dir, fname)

            shutil.copy(img_src, os.path.join(img_dest, fname))
            shutil.copy(lbl_src, os.path.join(lbl_dest, fname))

    print(f"Dataset split complete. {len(train_files)} to train, {len(test_files)} to test.")


if __name__ == "__main__":
    # dir = '/Users/ieo7086/Documents/tests/file_renaming_test/data'
    parser = argparse.ArgumentParser(description="Split images and labels into train and test.")
    parser.add_argument('--data_dir', required=True, help='Directory containing images and labels')
    parser.add_argument('--output_dir', required=True, help='Directory to save the split dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    random.seed(args.seed)

    split_dataset(args.data_dir, args.output_dir)
