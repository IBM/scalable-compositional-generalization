#!/bin/bash

# get command line args
dataset=$1
experiment=$2
model=${3:-"all"} # default to "all" if not provided

# set other experiment parameters
n_runs=4
split=composition
difficulty=0.1

# define possible attribute combinations
if [ "$dataset" = "dsprites" ]; then
    combos=("scale_shape")
elif [ "$dataset" = "iraven" ]; then
    combos=("size_type" "color_type" "size_color")
elif [ "$dataset" = "mpi3d" ]; then
    combos=("shape_size" "color_shape" "color_size" "color_height" "color_bgcolor" "shape_height" "shape_bgcolor" "size_height" "size_bgcolor" "height_bgcolor")
elif [ "$dataset" = "shapes3d" ]; then
    combos=("wall_floor" "wall_object" "wall_scale" "wall_shape" "floor_object" "floor_scale" "floor_shape" "object_scale" "object_shape" "scale_shape")
elif [ "$dataset" = "cars3d" ]; then
    combos=("elevation_type" "orientation_type" "orientation_elevation")
fi

# define list of models to test
pretrained=("resnet101_pretrained" "densenet_pretrained" "resnet152_pretrained")
resnet=("resnet18" "resnet34" "resnet50" "resnet101" "resnet152")
densenet=("densenet121" "densenet201" "densenet161")
swin=("swin_tiny" "swin_base")
convnext=("convnext_small" "convnext_base")
others=("vit" "mlp" "densenet")
ed=("ed")
if [ "$model" = "all" ]; then
    models=("${pretrained[@]}" "${resnet[@]}" "${ed[@]}" "${densenet[@]}" "${swin[@]}" "${convnext[@]}" "${others[@]}")
else
    models=("${model}")
fi

# execution sweep
for model in "${models[@]}"; do
    for attr in "${combos[@]}"; do
        for seed in $(seq 1 $n_runs); do
            python main.py --experiment-cfg configs/experiments/${experiment}.yml --seed=$seed \
            --data-cfg configs/datasets/${dataset}.yml --model-cfg configs/models/${model}.yml \
            data.training.targets=$attr data.training.split_attributes=$attr data.training.c=1 data.testing.c=1 \
            data.training.split=$split data.training.split_difficulty=$difficulty training.n_epoch=5 
        done
    done
done
