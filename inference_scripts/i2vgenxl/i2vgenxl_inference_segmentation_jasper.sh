python inference.py \
--model_name "i2vgenxl" \
--control_types "segmentation" \
--segmentation_type "odise" \
--local_checkpoint_path "checkpoints/adapter_jasper" \
--eval_input_type "frames" \
--evaluation_input_folder "assets/evaluation/frames" \
--global_step 70000 \
--n_sample_frames 16 \
--n_ref_frames 1 \
--num_inference_steps 50 \
--control_guidance_end 0.8 \
--use_size_512 false \
--height 160 \
--width 256 \
--evaluation_prompt_file "captions_AD_simul.json"


