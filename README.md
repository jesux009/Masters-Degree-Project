# Masters-Degree-Project: Segmentation of contrails in satellite imagery through machine learning techniques

This repository holds a compilation of the notebooks and files used for the completion of the End of Master's Degree Project "Segmentation of Contrails in Satellite Imagery through Machine Learning Techniques"
Four main types of components can be identified:

- **Dataset inference**: The file "OpenContrails_EDA" is a notebook used to explore the original dataset, find positive examples, soft labels, generate images, etc.
- **Network classes**: In this case the UNETv2.py can be inspected within the "Development" files, along with the individual backbones for training and validation. Variations with self-attention mechanisms were developed using the libraries "Segmentation Models Pytorch" and "TIMM"
- **Training and validation notebooks**: The actual files used for training and validation were developed in Kaggle to make use of its Graphical Processing Units instead of my local hardware

All trained networks are saved in the collection related to the project (https://www.kaggle.com/work/collections/13995497).

For more information, do not restrain from further contact at jesus.morales.lopez.27@gmail.com
