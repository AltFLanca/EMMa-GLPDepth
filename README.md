Inference run with command:

```
uv run src/inference.py --dataset-root /home/user/datasets/EMMa --asins /home/user/datasets/EMMa-ASINs/augmented_asins.npy --output /home/user/datasets/EMMa-GLP-Depth/Normalized/glp_depth_uint16_2.63.h5 --ckpt /home/user/models/glp-depth/normalized/epoch_20_model.ckpt --max-depth 2.63 --max-depth-eval 2.63 --storage uint16 
```

or

```
uv run src/inference.py --dataset-root /home/user/datasets/EMMa --asins /home/user/datasets/EMMa-ASINs/augmented_asins.npy --output /home/user/datasets/EMMa-GLP-Depth/Corrected/glp_depth_uint16_2.63.h5 --ckpt /home/user/models/glp-depth/corrected/epoch_25_model.ckpt --max-depth 2.63 --max-depth-eval 2.63 --storage uint16 
```

Then:

```
uv run src/valid_output_filtering.py --dataset-root /home/user/datasets/EMMa --augmentation-file /home/user/datasets/EMMa-GLP-Depth/Normalized/glp_depth_uint16_2.63.h5
```

or

```
uv run src/valid_output_filtering.py --dataset-root /home/user/datasets/EMMa --augmentation-file /home/user/datasets/EMMa-GLP-Depth/Corrected/glp_depth_uint16_2.63.h5
```