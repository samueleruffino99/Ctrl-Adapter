python inference.py \
--model_name "i2vgenxl" \
--control_types "segmentation" \
--segmentation_type "ade" \
--local_checkpoint_path "checkpoints/adapter_ade_360x640" \
--eval_input_type "frames" \
--evaluation_input_folder "assets/evaluation/frames" \
--global_step 100000 \
--n_sample_frames 16 \
--output_fps 12 \
--n_ref_frames 1 \
--num_inference_steps 50 \
--control_guidance_end 0.8 \
--use_size_512 false \
--height 360 \
--width 640 \
--evaluation_prompt_file "captions_segm_nuscenes.json"


