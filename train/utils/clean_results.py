import os
import re  # Import the re module
import shutil
from pathlib import Path

import wandb

# get current dir:
current_dir = Path(os.getcwd())
result_folder = Path(current_dir, "results")

# Define the regular expression pattern
pattern = r'\d+h-\d+m-\d+s'


def delete_wandb_run(run_name):
    user_input = input(
        f"Do you want to delete {run_name} on wandb (Y/n): ")
    if user_input.lower() == "y" or user_input == "":
        api = wandb.Api()
        runs = api.runs("collision_avoidance")
        # Run name you're looking for
        target_run_name = run_name

        # Search through runs in the project
        id = None
        for run in runs:
            if run.name == target_run_name:
                print("Found run ID:", run.id, "and deleting it")
                id = run.id
                run.delete()
                break
        if id is None:
            print("No wandb run found with name: ", run_name)


# iterate through all sub dirs of results folder
def clean_folder(parent: Path):
    if parent.name == "checkpoints" or parent.name == "wandb":
        # this one is allowed to be empty
        return

    for sub_content in parent.iterdir():
        if sub_content.is_file():
            continue
        # check if contains a folder named eval_videos
        if sub_content.name == "eval_videos" or sub_content.name == "logging":
            # check if sub_dir is empty
            if not list(sub_content.iterdir()):
                # take user input to delete the folder
                user_input = input(
                    f"Delete {parent.parent.name}/{parent.name} folder, {sub_content.name} contains no files? (Y/n): ")
                if user_input.lower() == "y" or user_input == "":
                    # delete the parent folder
                    shutil.rmtree(str(parent))
                    delete_wandb_run(parent.parent.name + "/" + parent.name)

                return
        else:
            # recursively call the function
            clean_folder(sub_content)

    # Use re.match to check if the folder name matches the regular expression pattern

    list_dir = list(parent.iterdir())
    main_folder = re.match(pattern, parent.name)
    # check if parent folder is empty
    if not list_dir:
        user_input = input(f"Delete {parent.name} folder, is now empty? (Y/n): ")
        if user_input.lower() == "y" or user_input == "":
            shutil.rmtree(str(parent))
            if main_folder:
                delete_wandb_run(parent.parent.name + "/" + parent.name)
    elif main_folder:
        valid_folders = False
        # we are in a main folder. Check if it contains subfolders
        for sub_dir in list_dir:
            if sub_dir.name == "eval_videos" or sub_dir.name == "logging":
                valid_folders = True
                break
        if not valid_folders:
            folder_names = [sub_dir.name for sub_dir in list_dir]
            user_input = input(f"Delete {parent.name} folder, no relevant subfolders found: {folder_names}? (Y/n): ")
            if user_input.lower() == "y" or user_input == "":
                shutil.rmtree(str(parent))
                delete_wandb_run(parent.parent.name + "/" + parent.name)


clean_folder(result_folder)
