# example script for test

scale=2
file=example_data/KiTS19_00000.nii.gz
python run.py CuNeRFx$scale --cfg configs/example.yaml --mode test --file $file --resume_type psnr --scales 1 2 --zpos -0.1 0.1 --angles 0 360 --axis 1 1 0 --asteps 45 --save_map --is_details --is_gif --is_video
