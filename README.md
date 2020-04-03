# Diagnosis for knee magnetic resonance imaging using deep convolutional neural network based on MRNet dataset



## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Screenshots](#screenshots)

## General info
Four types of CNN model created to detect general abnormalities and specific diagnoses (anterior cruciate ligament tears and meniscal tears) on knee MRI exams published by Stanford ML Group.

## Technologies
* Python
* PyCharm Community Edition
* Anaconda (Jupyter Notebook)
* PLGrid
* PyTorch
* Torch
* Tensorboard
* Numpy
* Pandas
* Scikit-learn
* Matplotlib

## Screenshots
<div>
  <p align="center">
    <img src="./images/seq3models.png" height="25%" width="25%">
  </p>
</div>

<div style="display: flex; flex-direction: row">
<div>
  <img height="25%" width="25%" src="./images/basic-06-3.png">
</div>

<div style="display: flex;">
  <img height="25%" width="25%" src="./images/relu-05-3.png">
  <img height="25%" width="25%" src="./images/relu-06-3.png">
 </div>
<div style="display: flex; flex-direction: column">
  <img height="25%" width="25%" src="./images/distBiasReLU.png">
  <img height="25%" width="25%" src="./images/distGradReLU.png">
  <img height="25%" width="25%" src="./images/distWeightReLU.png">
 </div>
 <div style="display: flex;">
  <img height="25%" width="25%" src="./images/histBiasReLU.png">
  <img height="25%" width="25%" src="./images/histGradReLU.png">
  <img height="25%" width="25% "src="./images/histWeightReLU.png">
</div>

<div style="display: flex;">
  <img height="25%" width="25%" src="./images/leaky-05-3.png">
  <img height="25%" width="25%" src="./images/leaky-06-1.png">
 </div>
<div style="display: flex;">
  <img height="25%" width="25%" src="./images/distBiasLeaky.png">
  <img height="25%" width="25%" src="./images/distGradLeaky.png">
  <img height="25%" width="25%" src="./images/distWeightLeaky.png">
 </div>
 <div style="display: flex;">
  <img height="25%" width="25%" src="./images/histBiasLeaky.png">
  <img height="25%" width="25%" src="./images/histGradLeaky.png">
  <img height="25%" width="25% "src="./images/histWeightLeaky.png">
</div>


<div>
  <p align="center">
    <img src="./images/trainACCepoch.png" height="25%" width="25%" >
  </p>
</div>
<div>
  <p align="center">
    <img src="./images/trainLOSScomp.png" height="25%" width="25%">
  </p>
</div>
