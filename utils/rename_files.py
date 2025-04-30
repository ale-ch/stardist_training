import os

def rename_files_in_folder(folder_path):
    """
    Renameth all files in the given folder that contain '.ome' afore their extension,
    such that '.ome' is removéd. For example: 'dapi.ome.tif' becometh 'dapi.tif'.
    """
    for filename in os.listdir(folder_path):
        if '.ome.' in filename:
            new_name = filename.replace('.ome.', '.')
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renaméd: {filename} -> {new_name}")


if __name__ == "__main__":
    dir = '/Users/ieo7086/Documents/tests/file_renaming_test/data/images'
    rename_files_in_folder(dir)
