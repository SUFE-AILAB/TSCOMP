import os
import re
import shutil

checkpoint_dir = 'checkpoints'

if not os.path.exists(checkpoint_dir):
    print(f"Directory {checkpoint_dir} does not exist.")
    exit(1)

def rename_pth_in_folder(target_path):
    if not os.path.exists(target_path):
        return
    for file_name in os.listdir(target_path):
        if file_name.endswith('.pth') and file_name != 'checkpoint.pth':
            src = os.path.join(target_path, file_name)
            dst = os.path.join(target_path, 'checkpoint.pth')
            print(f"Renaming {file_name} to checkpoint.pth in {target_path}")
            try:
                os.rename(src, dst)
            except Exception as e:
                print(f"Error renaming {file_name} in {target_path}: {e}")

# Regex to match the pattern
pattern = re.compile(r'sl(\d+)_ll(\d+)_pl(\d+)')

for folder_name in sorted(os.listdir(checkpoint_dir)):
    folder_path = os.path.join(checkpoint_dir, folder_name)
    
    # Skip files and skip directories that are already organized (like 96_0_96)
    if not os.path.isdir(folder_path):
        continue
        
    # If the folder name is already in the target format (digits_digits_digits), check its subfolders
    if re.fullmatch(r'\d+_\d+_\d+', folder_name):
        for sub_folder_name in os.listdir(folder_path):
             sub_folder_path = os.path.join(folder_path, sub_folder_name)
             if os.path.isdir(sub_folder_path):
                 rename_pth_in_folder(sub_folder_path)
        continue

    match = pattern.search(folder_name)
    if match:
        sl = match.group(1)
        ll = match.group(2)
        pl = match.group(3)
        
        target_folder_name = f"{sl}_{ll}_{pl}"
        target_folder_path = os.path.join(checkpoint_dir, target_folder_name)
        
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)
            print(f"Created directory: {target_folder_path}")

        # Extract parameters for renaming
        # Format: LTF_{args.data_name}_{args.model}_dmodel{args.d_model}_epoch{args.train_epochs}
        new_folder_name = folder_name
        
        dm_match = re.search(r'_dm(\d+)', folder_name)
        epochs_match = re.search(r'_epochs(\d+)', folder_name)

        if dm_match and epochs_match:
            d_model = dm_match.group(1)
            train_epochs = epochs_match.group(1)
            
            parts = folder_name.split('_')
            # User specified: model is 2nd value (index 1), data_name is 3rd value (index 2)
            # Special handling for Nonstationary_Transformer which has an extra underscore
            if len(parts) >= 3:
                model = parts[1]
                data_name = parts[2]
                
                if model == "Nonstationary" and data_name == "Transformer":
                     model = "Nonstationary_Transformer"
                     data_name = parts[3]
                
                new_folder_name = f"LTF_{data_name}_{model}_dmodel{d_model}_epoch{train_epochs}"# 

        destination = os.path.join(target_folder_path, new_folder_name)
        
        print(f"Moving {folder_name} to {destination}")
        try:
             shutil.move(folder_path, destination)
             rename_pth_in_folder(destination)
        except Exception as e:
            print(f"Error moving {folder_name}: {e}")

print("Organization complete.")
