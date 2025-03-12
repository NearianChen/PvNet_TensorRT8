import os
import random

def delete_random_files(folder_path, num_files_to_delete):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    random.shuffle(files)
    files_to_delete = files[:num_files_to_delete]
    
    for file in files_to_delete:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted {file}")

folder_path = 'data/new_nocrop_0116/rgb'
num_files_to_delete = 30
delete_random_files(folder_path, num_files_to_delete)
