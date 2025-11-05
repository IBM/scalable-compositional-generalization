# Data extraction
All the datasets used in this work are publicly available online.
Here we report how the single datasets were generated.

### I-RAVEN
First, download the data:
```bash
git clone https://github.com/husheng12345/SRAN
conda create -n iraven
conda activate iraven
(iraven) pip2 install --user -r SRAN/I-RAVEN/requirements.txt
python2 SRAN/I-RAVEN/main.py --save-dir ./data/I-RAVEN
```
Then run
```bash
python visgen/datasets/generation/iraven_extraction.py --data_path ./data/I-RAVEN
```
to extract the targets

### dSprites
```bash
wget https://github.com/google-deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true
mv  'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true' dsprites.npz
```

### MPI3D
```bash
wget https://storage.googleapis.com/mpi3d_disentanglement_dataset/data/real.npz
```

### 3D Shapes
```bash
curl -X GET -o "3dshapes.h5" "https://storage.googleapis.com/storage/v1/b/3d-shapes/o/3dshapes.h5?alt=media"
```

### 3D Cars
```bash
pip install gdown
gdown 1aCo9wD4kbY4V0cu7qFXSHF4rmJlEb_E-
unzip cars3d.zip
```

### CLEVR
We generated a dataset of 100k images using the original CLEVR generative code from https://github.com/facebookresearch/clevr-dataset-gen, on `blender-2.78c`, using the command
```bash
blender --background --python render_images.py -- --num_images 100000 --min_objects 1 --max_objects 1 --render_tile_size 16
```
