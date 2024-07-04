from utils import * 
# import shutil
import shutil

# list all the dir names in input_dir directory
intput_dir = "/mnt/d/marketing/random"
move_destinaton = "/mnt/d/marketing/video/prescan_augmented"

dirs = os.listdir(intput_dir)[::-1]

for dir in dirs:

    # make dir in move_destination
    destination = os.path.join(move_destinaton, dir.replace("_random_speed", "").replace("_","-"))

    # move the seg_ODISE folder to the destination
    shutil.move(os.path.join(intput_dir, dir, "seg_ODISE"), destination)




