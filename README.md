# Enhancing Data Efficiency with a Trustworthy Counterfactual Generative Model
This is official repository for paper Enhancing Data Efficiency with a Trustworthy Counterfactual Generative Model

The structure of the model is:

![image](https://github.com/yza2542/Causal-DiffuseVAE/blob/main/resource/structure.png)

The shadow datasets could be downloaded from https://drive.google.com/file/d/1dnDRICw9kkLB8NNkCAQJmwda6D_Sos21/view?usp=sharing and https://drive.google.com/file/d/1BFfnrnEEIZPTZwu-41AE_kV7yv69qwEx/view?usp=sharing.

The results of the shadow datasets and CelebA dataset are shown as:
![image](https://github.com/yza2542/Causal-DiffuseVAE/blob/main/resource/result_shadow.png)
![image](https://github.com/yza2542/Causal-DiffuseVAE/blob/main/resource/result_celeba.png)


To train the VAE model, the following code is used:

For shadow Datasets
```
sh train_cvae_shadow.sh
``` 

For Synthetic and CelebA Dataset
```
sh train_cvae.sh
``` 

To train the DDPM model, the following code is used:
```
sh train_ddpm.sh
```

To generate the counterfactual images, the following code is used:
```
sh test_inference_ddpm.sh
```
