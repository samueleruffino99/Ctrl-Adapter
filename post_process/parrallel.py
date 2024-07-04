import os
from multiprocessing import Pool
from PIL import Image
from utils import process_images, map_rgb, create_video_from_frames

def process_directory(dir):
    try:
        base_path = "/mnt/d/marketing/process"
        types = ["rgb", "seg", "seg_ADE", "seg_ODISE"]
        
        for type in types:
            try:
                input_dir = os.path.join(base_path, dir, type)
                output_dir = os.path.join(base_path, dir, f"{type}_70")

                if "seg_ADE" in type or "seg_ODISE" in type:
                    continue

                fov_from = 120  # Initial FOV in degrees
                fov_to = 94  # Desired FOV in degrees

                adjust_fov_method = type not in ["seg", "diff"]
                process_images(input_dir, output_dir, fov_from, fov_to, adjust_fov_method)

            except Exception as e:
                print(f"Error processing {type} in {dir}: {e}")
        
        # Segmentation
        try:
            path = os.path.join(base_path, dir, "seg_70")
            output_path_odise = os.path.join(base_path, dir, "seg_ODISE")
            output_path_ade = os.path.join(base_path, dir, "seg_ADE")

            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".png"):
                        im = Image.open(os.path.join(root, file))

                        destination = output_path_odise
                        if not os.path.exists(destination):
                            os.makedirs(destination)

                        output_file_path = os.path.join(destination, file)
                        if not os.path.exists(output_file_path):
                            mapped_image = map_rgb(im, "odise")
                            mapped_image.save(output_file_path)

                        destination = output_path_ade
                        if not os.path.exists(destination):
                            os.makedirs(destination)

                        output_file_path = os.path.join(destination, file)
                        if not os.path.exists(output_file_path):
                            mapped_image = map_rgb(im, "ade")
                            mapped_image.save(output_file_path)

                        im.close()
        except Exception as e:
            print(f"Error during segmentation in {dir}: {e}")

        # Video creation
        # try:
        #     for type in ["rgb", "seg"]:
        #         type = f"{type}_70"
        #         input_dir = os.path.join(base_path, dir, type)
        #         frame_rate = 10  # frames per second
        #         video_length = 1.8  # seconds
        #         output_video_path = os.path.join(base_path, dir, f"output_video_{type}.avi")

        #         create_video_from_frames(input_dir, output_video_path, frame_rate, video_length)
        # except Exception as e:
        #     print(f"Error creating video for {dir}: {e}")

    except Exception as e:
        print(f"Error processing directory {dir}: {e}")

if __name__ == "__main__":
    base_path = "/mnt/d/marketing/process"
    dirs = os.listdir(base_path)[::-1]

    with Pool() as pool:
        pool.map(process_directory, dirs)
