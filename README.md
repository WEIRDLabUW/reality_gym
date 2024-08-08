# Reality Gym
Reality Gym for URDFormer. If you want to try converting images into URDFs, check out [URDFormer](https://github.com/WEIRDLabUW/urdformer) and [Website](https://urdformer.github.io/).

This repo contains (a) URDF Assets from internet images (b) Mesh and texture randomization.

First, download and unzip the assets created from the internet images [link](https://drive.google.com/file/d/1LQkSmvhwAIGVCN4eHNJAM7BR6EyTpFma/view?usp=sharing). After this step, you repo should look like:
```bash
reality_gym/
├── assets/
│   ├── all_meshes/ (this contains all predefined geometries for doors, drawer, handles etc.)
│   └── cabinets/ (this contains details for each asset)
│       ├── images/ 
│       ├── labels/ (this contains GT labels for each asset, including bbox, class, positions, scales etc)
│       ├── textures/
│       ├── urdfs/
│   └── dishwashers
│       └── ...
│   └── fridges
│       └── ...
│   └── kitchens
│       └── ...
│   └── ovens
│       └── ...
│   └── washers
│       └── ...
│   └── partnet
...
```
To visualize kitchen GT assets, run:
```bash
python gt_demo.py --scene kitchens --texture
```


To visualize objects GT assets, run:
```bash
gt_demo.py --scene objects --texture
```

