# Brain-Diffuser
Official repository for the paper ["**Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion**"](https://arxiv.org/abs/2303.05334) by Furkan Ozcelik and Rufin VanRullen.

## Results
The following are a few reconstructions obtained : 
<p align="center"><img src="./figures/Reconstructions.png" width="600" ></p>

Code will be available soon!

## Instructions 

### Data Acquisition and Processing

1. Download NSD data from NSD AWS Server:
    ```
	cd data
	python download_nsddata.py
	```
2. Download "COCO_73k_annots_curated.npy" file from ["HuggingFace NSD"](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main)
3. Prepare NSD data for the Reconstruction Task:
    ```
	cd data
	python prepare_nsddata.py -sub 1
    python prepare_nsddata.py -sub 2
    python prepare_nsddata.py -sub 5
    python prepare_nsddata.py -sub 7
	```