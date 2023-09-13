import os
import pandas as pd
import random

def pick_random_images(directory, num_images=100):
    # Get all files in directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Filter for image files (based on extension). You can add more extensions if needed.
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # Pick random image files
    selected_images = sorted(random.sample(image_files, min(num_images, len(image_files))))
    
    # Create DataFrame
    df = pd.DataFrame({
        "ImagePath": selected_images,
        "ManualPumpingCount": [None] * len(selected_images)
    })
    
    return df

# Test
directory = '/mnt/external.data/TowbinLab/kstojanovski/20220401_Ti2_20x_160-182-190_pumping_25C_20220401_173300_429/analysis/str_pharynx_videos/' # replace this with the path to your image directory
df = pick_random_images(directory)

# Save to CSV
df.to_csv('pumping_manual_old_experiment.csv', index=False) # replace this with the path to your output CSV file
