# README

## Steps

## ENV
Due to the conflicts between a61 and PC, so firstly modify the environment name before 
install the environment.  
**new environment name: control_pc / control_a61**  
`conda env create -f environmetn.yaml`

## Code of training
1. First Step: 

Following the steps of tutorials and train.md in folder docs,
the dataset will be changed into the Quilt1M. This step is also called 
pretraining controlnet with Quilt1M dataset.


2. Second Step

Finetuning on the Chaoyang dataset, Remember a thing that using the 
different text prompts.

3. Third Step

When generating enough data, using the pretrained classification model 
to do the classification. Please notice that how to arrange the dataset
everytime, like those model with similar data augmentation.

4. Fourth Step

Evaulating the model. 


## TODO
- [ ] Using the pretrained stable diffusion model as listed on the 
huggingface to pretrain on the Quilt1M dataset. 
- [ ] Using the pretrained model I have trained with Quilt1M dataset to 
train the ControlNet dataset.
- [x] Generate the edge maps and the json file for training. If I use the 
quilt1M for pretraining, the training and testing are OK for usage. 
- [ ] Coding for training the ControlNet with Quilt1M.
- [ ] Coding for training the Chaoyang Dataset.
