import argparse
import os
import matplotlib.pyplot as plt
import tifffile as tiff

def main():
    parser = argparse.ArgumentParser(description="Run StarDist2D prediction on a test image.")
    parser.add_argument('--preds_dir', required=True, help='Path to directory with tiff predicted masks.')
    parser.add_argument('--gt_dir', required=True, help='Path to directory with tiff groud truth masks.')
    parser.add_argument('--outdir', required=True, help='Directory to save quality control plots.')

    args = parser.parse_args()

    preds_dir = args.preds_dir
    gt_dir = args.gt_dir
    outdir = args.outdir

    preds_files = [os.path.join(preds_dir, file) for file in os.listdir(preds_dir)]
    masks_files = [os.path.join(gt_dir, file) for file in os.listdir(gt_dir)]

    print(f"Found {len(preds_files)} predicted masks and {len(masks_files)} ground truth masks.")

    for pred_file, mask_file in zip(preds_files, masks_files):
        filename = f"{os.path.basename(pred_file).split('.')[0]}.jpg"
        output_path = os.path.join(outdir, filename)
        print(f"Processing {pred_file} and {mask_file}...")
        mask = tiff.imread(mask_file)
        pred = tiff.imread(pred_file)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(mask > 0, alpha=0.3, cmap='Blues')
        plt.imshow(pred > 0, cmap='Reds', alpha=0.2)
        print(f"Saving plot to {output_path}...")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}.")


if __name__ == '__main__':
    main()
