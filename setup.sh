# create conda env
conda create -n cunerf python=3.9 -y

# install Pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# install other packages
pip install gpustat\
            simpleitk\
            pillow\
            scikit-image==0.19.2\
            numpy\
            tqdm\
            pyyaml\
            lpips -i https://pypi.douban.com/simple