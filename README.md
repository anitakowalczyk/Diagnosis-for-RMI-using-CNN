# Diagnosis for knee magnetic resonance imaging using deep convolutional neural network based on MRNet dataset

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

<h2>Models layers</h2>
<div style="margin-top: 30px">
  <p align="center">
    <img src="./images/seq3models.png" height="50%" width="50%">
  </p>
</div>

<h2>Model result comparision</h2>
<ul>
  <li>basic model - learning rate 0.6 - patience 3</li>
  <li>relu model - learning rate 0.5 - patience 3</li>
  <li>relu model - learning rate 0.6 - patience 3</li>
</ul>
<p align="center" >
  <img style="display: inline-block;" height="30%" width="30%" src="./images/basic-06-3.png">
  <img style="display: inline-block;" height="30%" width="30%" src="./images/relu-05-3.png">
  <img style="display: inline-block;"height="30%" width="30%" src="./images/relu-06-3.png">
 </p>
 
<h3>Empirical distributions of bias, gradient and weight</h3>
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/distBiasReLU.png">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/distGradReLU.png">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/distWeightReLU.png">
 </p>

<h3>Histograms of bias, gradient and weight</h3> 
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/histBiasReLU.png">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/histGradReLU.png">
  <img style="display: inline-block;" height="33%" width="33% "src="./images/histWeightReLU.png">
</p>

<h2>Model result comparision</h2>
<ul>
  <li>leaky model - learning rate 0.5 - patience 3</li>
  <li>leaky model - learning rate 0.6 - patience 1</li>
</ul>
<p align="center" style="margin-top: 300px;">
  <img style="display: inline-block;" height="40%" width="40%" src="./images/leaky-05-3.png">
  <img style="display: inline-block;" height="40%" width="40%" src="./images/leaky-06-1.png">
 </p>
 
 <h3>Empirical distributions of bias, gradient and weight</h3>
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/distBiasLeaky.png">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/distGradLeaky.png">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/distWeightLeaky.png">
 </p>
 
 <h3>Histograms of bias, gradient and weight</h3> 
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/histBiasLeaky.png">
  <img style="display: inline-block;" height="33%" width="33%" src="./images/histGradLeaky.png">
  <img style="display: inline-block;" height="33%" width="33% "src="./images/histWeightLeaky.png">
</p>

<h2>Models accuracy and loss</h2>
<p align="center" style="margin-top: 300px;">
  <img style="display: inline-block;" src="./images/trainACCepoch.png" height="50%" width="50%" >
  <img style="display: inline-block;" src="./images/trainLOSScomp.png" height="50%" width="50%">
</p>

