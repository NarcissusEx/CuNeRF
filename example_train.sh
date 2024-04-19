# example script for training CuNeRF

scale=2
file=example_data/KiTS19_00000.nii.gz
python run.py CuNeRFx$scale --cfg configs/example.yaml --scale $scale --mode train --file $file --save_map --resume