## Unsupervised Contrastive Learning of Image Representations from Ultrasound Videos with Hard Negative Mining

This is the official implementation of the UCL-method based on Hard Negative Mining, introduced in our paper [https://arxiv.org/abs/2207.13148](https://arxiv.org/abs/2207.13148). 

### Installation

Our code is tested on Python 3.7 and Pytorch 1.3.0, please install the environment via 

```
pip install -r requirements.txt
```

We ran some of our baselines using [lightly](https://github.com/lightly-ai/lightly).

### Our Datasets
#### Video Dataset
To get our Video Dataset(GBUSV) follow the instructions [here](https://gbc-iitd.github.io/data/gbusv).

#### Image Dataset
To get our Image Dataset(GBCU) follow the instructions [here](https://gbc-iitd.github.io/data/gbcu).

### Model Zoo 

We provide the models pretrained on for 50 epochs and the fine-tuned classifier. 

| Method        | Downstream Task | Backbone | Pre-trained model | Fine-tuned model |
|---------------|:--------:|:--------:|:-----------------:|:----------------:|
| UCL   | Gallbladder Cancer Detection | Resnet18 | [pretrain ckpt](https://add-pretrain)  |  [classifier ckpt](https://add-finetune)  |
| UCL   | COVID-19 Detection         | Resnet18 | [pretrain ckpt](https://add-pretrain)  |  [classifier ckpt](https://add-finetune)  |
| UCL   | Gallbladder Cancer Detection | Resnet50 | [pretrain ckpt](https://add-pretrain)  |  [classifier ckpt](https://add-finetune)  |
| UCL   | COVID-19 Detection         | Resnet50 | [pretrain ckpt](https://add-pretrain)  |  [classifier ckpt](https://add-finetune)  |


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

#### Note on flags in pree-training script
In the pre-training script some flags which can be experimented if the default configuration doesn't give results:
1. moco-t: It selects the temperature used while loss calculation
2. soft-nn-t: It selects the temperature used to calculate cross video negative
3. single-loss-ncap-support-size: It decides the number of cross video negatives used to form cross video negative used in the loss function
4. num-negatives: It selects ths number of intra video negatives to be used

### Experiments on GB Datasets

#### Unsupervised Contrastive Pretrain
```
bash scripts/pretrain_gb.sh
```
#### Fine-tune for Downstream Classification
```
bash scripts/finetuning_gb.sh <split_no> <model_path> <checkpoint_no> <gpu_id>
```
#### Evaluate Classifier 
```
bash scripts/evaluate_gb.sh <split_no> <model_path> <checkpoint_no> <gpu_id>
```
### Experiments on POCUS and Butterfly Dataset

#### Getting Butterfly and POCUS Dataset
Please follow instructions on [USCL Repo](https://github.com/983632847/USCL) to get these datasets.

#### Unsupervised Contrastive Pretrain
1. After Downloading the butterfly dataset make sure the frames in each video follow the naming convention as described in Data Preparation section of this README.
2. Modify the num_var variable in Line 165 of [Main Script](train_ucl.py) to 22.
3. Run the command:
``` 
bash scripts/pretrain_ucl_butterfly.sh
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
