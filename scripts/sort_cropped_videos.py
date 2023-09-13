import os
import shutil
from joblib import Parallel, delayed

def copy(filename, source_folder, destination_folder, video_extension):
    if filename.endswith(video_extension):
        # Determine the target folder based on the video extension
        target_folder = destination_folder

        if "_pharynx" in filename:
            target_folder = os.path.join(destination_folder, "cropped_pharynx_videos")
        elif "_body" in filename:
            target_folder = os.path.join(destination_folder, "cropped_body_videos")

        # Copy the video to the target folder
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(target_folder, filename)
        shutil.copy(source_path, destination_path)
        print(f"Copied {filename} to {target_folder}")
    
def copy_videos(source_folder, destination_folder, video_extension):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    Parallel(n_jobs=-1, prefer="threads")(delayed(copy)(filename, source_folder, destination_folder, video_extension) for filename in os.listdir(source_folder))
    
            

# Example usage
source_folder = "/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/analysis/optimized_cropped_videos/"
destination_folder = "/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/analysis/"
video_extension = ".tiff"

copy_videos(source_folder, destination_folder, video_extension)
