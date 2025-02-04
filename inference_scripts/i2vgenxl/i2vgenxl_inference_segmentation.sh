python inference.py \
--model_name "i2vgenxl" \
--control_types "segmentation" \
--huggingface_checkpoint_folder "i2vgenxl_multi_control_adapter" \
--eval_input_type "frames" \
--evaluation_input_folder "assets/evaluation/frames" \
--n_sample_frames 16 \
--n_ref_frames 1 \
--num_inference_steps 50 \
--control_guidance_end 0.8 \
--use_size_512 false \
--height 224 \
--width 224 \
--evaluation_prompt_file "captions_multi.json" 

