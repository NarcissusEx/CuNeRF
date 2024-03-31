# example script for test

scale=2
file=example_data/KiTS19_00000.nii.gz
python run.py CuNeRFx$scale --cfg configs/example.yaml --mode test --file $file --resume_type psnr --scale_init 1 --scale_final 2 --z_init -0.1 --z_final 0.1 --angle_init 0 --angle_final 360 --axis 1 1 0 --asteps 45 --save_map --is_details --is_gif --is_video