# LTR

A general PyTorch based framework for learning tracking representations. 
## Table of Contents

## Quick Start
The installation script will automatically generate a local configuration file  "admin/local.py". In case the file was not generated, run ```admin.environment.create_default_local_file()``` to generate it. Next, set the paths to the training workspace, 
i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.  
```bash
conda activate pytracking
python run_training.py train_module train_name
```
Here, ```train_module``` is the sub-module inside ```train_settings``` and ```train_name``` is the name of the train setting file to be used.

quake  training 
```bash
python -m torch.distributed.launch --nproc_per_node 4 run_siamft.py
```


# train 
train2, siamft2: baseline: mca_neck+ mlp head   11-12
train2_smca_mlp, siamft2_smca_mlp: mca_smca_neck + mlp head.  11-12
train2_focal, siaft2_focal: mca_neck + focal head  14-15 

