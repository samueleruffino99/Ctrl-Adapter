accelerate launch train.py \
--yaml_file configs/i2vgenxl_train_segmentation_nuscenes.yaml \
--evaluation_input_folder "assets/evaluation/frames" \
--evaluation_output_folder "Output_i2vgenxl_segmentation_nuscenes" \
--evaluation_prompt_file "captions_segm_nuscenes.json" \
--num_inference_steps 50 \
--control_guidance_end 1.0 \
--max_train_steps 100000 \
--save_n_steps 5000 \
--validate_every_steps 5000 \
--save_starting_step 5000 \
--extract_control_conditions True 
