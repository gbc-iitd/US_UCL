## Unsupervised Contrastive Learning of Image Representations from Ultrasound Videos with Hard Negative Mining

This is the official implementation of the UCL-method based on Hard Negative Mining, introduced in our paper [https://arxiv.org/abs/2207.13148](https://arxiv.org/abs/2207.13148). 

### Installation

Our code is tested on Python 3.7 and Pytorch 1.3.0, please install the environment via 

```
pip install -r requirements.txt
```

Note: To run our code you'll need multiple GPUs this code is not designed to run on single GPU systems.    

### Our Datasets
1. Video Dataset: To get our Video Dataset(GBUSV) follow the instructions [here](https://gbc-iitd.github.io/data/gbusv).      
2. Image Dataset: To get our Image Dataset(GBCU) follow the instructions [here](https://gbc-iitd.github.io/data/gbcu).

### Model Zoo 

We provide the models pretrained on Video Data for 50 epochs.

| Method        | Downstream Task | Backbone | Pre-trained model | Fine-tuned model (on a single val split)
|---------------|:--------:|:--------:|:-----------------:|:-------------------------:|
| UCL   | Gallbladder Cancer Detection | Resnet50 | [pretrain ckpt](https://drive.google.com/file/d/1nu4-WtuUj7VIV4vyKmoz9M0Tw2GXvS9P/view?usp=sharing)  | [finetune ckpt](https://drive.google.com/file/d/1H9Abh9YvKkIUe38opbAPWhDEt51XJxTd/view) | 
| UCL   | COVID-19 Detection         | Resnet50 | [pretrain ckpt](https://drive.google.com/file/d/1giXcf52tD2zUmQuC_DXBXdCXXnS3xFIm/view?usp=sharing)  |  - |


### Run on Custom Dataset
#### Data preparation

The directory structure should be as followed:
```
Data_64
│   ├── Video_1 
│   │   ├── 00000.jpg
|   |   ├── 00001.jpg
|   |   ├── 00002.jpg
|   |   ├──  .....
|   ├── Video_2
│   │   ├── 00000.jpg
|   |   ├── 00001.jpg
|   |   ├── 00002.jpg
|   |   ├──  ....
```

#### Note on flags in pre-training script
In the pre-training script some flags which can be experimented if the default configuration doesn't give results:
1. moco-t: It selects the temperature used while loss calculation
2. soft-nn-t: It selects the temperature used to calculate cross video negative
3. single-loss-ncap-support-size: It decides the number of cross video negatives used to form cross video negative used in the loss function
4. num-negatives: It selects ths number of intra video negatives to be used

### Experiments on GB Datasets

#### Unsupervised Contrastive Pretrain
```
bash scripts/pretrain_gb.sh <Video_Data_path>
```
#### Fine-tune for Downstream Classification
```
bash scripts/finetuning_gb.sh <path_to_train_split_txt_file> <path_to_val_split_txt_file> <path_to_pretrained_model> <path_for_saving_finetuned_model> <gpu_id>
```
#### Evaluate Classifier 
```
bash scripts/evaluate_gb.sh <path_to_train_split_txt_file> <path_to_val_split_txt_file> <path_to_pretrained_model> <gpu_id>
```
### Experiments on POCUS and Butterfly Dataset

#### Getting Butterfly and POCUS Dataset
Please follow instructions on [USCL Repo](https://github.com/983632847/USCL) to get these datasets.

#### Unsupervised Contrastive Pretrain
1. After Downloading the butterfly dataset make sure the frames in each video follow the naming convention as described in Data Preparation section of this README.
2. Run the command:
``` 
bash scripts/pretrain_ucl_butterfly.sh <path_to_butterfly_data>
```

#### Fine-Tuning and Evaluating Classifier
For this we have followed the evaluation mechanism as proposed in [USCL](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_60), the script modified for our proposed model can be found [here](scripts/eval_ucl_pocus.py).     

Steps to run this script:    
1. Setup environment and code from [USCL repo](https://github.com/983632847/USCL).
2. Place [Our Evaluation Script](scripts/eval_ucl_pocus.py) in eval_pretrained_model directory of USCL repo.
3. Run the command:
```
python eval_ucl_pocus.py --path <model_path> --gpu <gpu_id>
```

## Acknowledgements
The codebase is based on [CycleContrast](https://github.com/happywu/CycleContrast) and [MoCo](https://github.com/facebookresearch/moco).
