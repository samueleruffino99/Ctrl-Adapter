python inference.py \
--model_name "svd" \
--control_types "segmentation" \
--huggingface_checkpoint_folder "svd_depth" \
--eval_input_type "frames" \
--evaluation_input_folder "assets/evaluation/frames" \
--skip_conv_in True \
--n_sample_frames 14 \
--extract_control_conditions True \
--num_inference_steps 25 \
--control_guidance_end 0.8 \
--use_size_512 false \
--height 128 \
--width 128 \
--evaluation_prompt_file "captions_AD_real.json" 

