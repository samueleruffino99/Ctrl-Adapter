from post_process.utils import *

# list all the dir names in input_dir directory
intput_dir = "/mnt/d/marketing/objectdetection"


dirs = os.listdir(intput_dir)[::-1]
dirs_two = os.listdir(intput_dir)[::-1]


types = ["rgb", "seg",   "seg_ADE" , "seg_ODISE" ]
base_path   = "/mnt/d/marketing/objectdetection"
for dir in dirs:
    for type in types:
        try:
            input_dir = os.path.join(base_path,dir, type)
            output_dir = os.path.join(base_path,dir, f"{type}_70")

            if "seg_ADE" in type or "seg_ODISE" in type:
                continue

            fov_from = 120  # Initial FOV in degrees
            fov_to = 94  # Desired FOV in degrees

            

            if type == "seg" or type == "diff":
                adjust_fov_method = False
            else:
                adjust_fov_method = True

            process_images(input_dir, output_dir, fov_from, fov_to, adjust_fov_method)
            
        except Exception as e:
            print(e)
            continue


# Segmenrat
for dir in dirs:
    try:
        # Get all the images in the segmentation folder
        path = os.path.join(base_path, dir, "seg_70")
        output_path_odise = os.path.join(base_path,dir, "seg_ODISE")
        output_path_ade = os.path.join(base_path, dir,"seg_ADE")

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".png"):
                    dir_name = root.split("/")[-1]
                    im = Image.open(os.path.join(root, file))

                    # Save the changed image
                    destination = output_path_odise
                    if not os.path.exists(destination):
                        os.makedirs(destination)

                    output_file_path = os.path.join(destination, file)
                    if os.path.exists(output_file_path):
                        continue

                    # Map RGB colors efficiently
                    mapped_image = map_rgb(im, "odise")

                    # Save the mapped image
                    mapped_image.save(output_file_path)
                    

                    ###############
                    # destination = output_path_ade
                    # if not os.path.exists(destination):
                    #     os.makedirs(destination)

                    # output_file_path = os.path.join(destination, file)
                    # if os.path.exists(output_file_path):
                    #     continue

                    # # Map RGB colors efficiently
                    # mapped_image = map_rgb(im, "ade")

                    # # Save the mapped image
                    # mapped_image.save(output_file_path)

                    # Close the image file
                    im.close()
    except Exception as e:
        print(e)
        continue


# for dir in dirs_two: 
#     for type in types:
#         try:

#             if type == "seg" or type == "rgb":
#                 type = type + "_70"
        

#             input_dir = os.path.join(base_path, dir, f"{type}")

#             frame_rate = 10  # frames per second
#             video_length = 1.8  # seconds
#             output_video_path = os.path.join(base_path,dir, f"output_video_{type}.avi")

#             # Ensure the output directory is writable
#             if not os.access(base_path, os.W_OK):
#                 print(f"Cannot write to directory: {base_path}")
#             else:
#                 create_video_from_frames(input_dir, output_video_path, frame_rate, video_length)
#         except Exception as e:
#             print(e)