# example script for test

scale=2
# file=data/train_0027_0000.nii.gz
# file=data/BRATS_014.nii.gz
# file=data/1-1_ZX4168211_t1_vibe_dixon_cor_W_cropped.nii.gz
file=data/1-1_ZX4168211_t1_vibe_dixon_cor_W_resized.nii.gz
# python run.py CuNeRFx$scale --cfg configs/example.yaml --mode test --file $file --resume_type psnr --scale_init 1 --scale_final 2 --z_init -0.1 --z_final 0.1 --angle_init 0 --angle_final 360 --axis 1 1 0 --asteps 45 --save_map --is_details --is_gif --is_video

# python run.py CuNeRFx$scale --cfg configs/example.yaml --mode test --file $file --resume_type psnr   --axis 1 1 0 --asteps 45   --modality T1w  --angles 0 360 --save_map --is_details --is_gif --is_video 
# python run.py CuNeRFx$scale --cfg configs/example.yaml --mode test --file $file --resume_type psnr --zpos -0.1 0.1  --axis 1 0 0 --asteps 45    --angles 0 360 --save_map --is_details --is_gif --is_video 
# python run.py CuNerf1121 --cfg configs/example.yaml --mode test --file $file  --scales 1 2 --zpos 0 --angles 0 360 --axis 1 0 0 --asteps 45 --save_map --is_details --is_gif --is_video --modality T1w

python run.py debug_CuNeRFx$scale --cfg configs/example.yaml --mode test --file $file --resume_type psnr --zpos -0.5 0.5  --axis 1 0 0 --asteps 45 --save_map --is_details --is_gif --is_video 
