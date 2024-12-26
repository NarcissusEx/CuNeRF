# example script for training CuNeRF

scale=2
# file=data/train_0027_0000.nii.gz
# file=data/BRATS_014.nii.gz
file=data/1-1_ZX4168211_t1_vibe_dixon_cor_W_resized.nii.gz
# python CuNeRF/run.py CuNeRFx$scale --cfg configs/example.yaml --scale $scale --mode train --file $file --save_map --resumeconda

python run.py ZX_CuNeRFx$scale --cfg configs/example.yaml --scale $scale --mode train --file $file --save_map 

