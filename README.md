# Machine Learning to Denoise Images for Better OCR Accuracy

This project is an adaptation of this tutorial and used only for learning purposes: https://www.pyimagesearch.com/2021/10/20/using-machine-learning-to-denoise-images-for-better-ocr-accuracy/#download-the-code

## Setting Up the project 🚀

First and foremost clone the project with: 
```
$ git clone https://github.com/AntonioBriPerez/Ocr-Denoiser
```
You don't need to extract the zip files in order to train the model. 

Once you have cloned the repository you will need to extract the features from the noisy images. This script will extract 5 x 5 - 25-d feature vectors and the it will extract the target (or cleaned) pixel value from the correspondiente ground truth standard image. And then, this features will be saved in a csv file (~200MB). To extract this features you will have to execute: 
```
$ python3 build_features.py
```
It will generate the following output: 

![alt text](https://github.com/AntonioBriPerez/Ocr-Denoiser/blob/main/readme_images/extract_features.png)

Once you have done that we will have to load those features in a proper split to train our Random Forest Regressor. That code is implemented in the file train_denoiser.py. To train the model you will have to run the command:

```
$ python train_denoiser.py
```
And it will generate: 
![alt text] (https://github.com/AntonioBriPerez/Ocr-Denoiser/blob/main/readme_images/train_denoiser.png)

To check that the model performs good you can execute: 
```
$ python3 denoise_document.py --testing denoising-dirty-documents/test
```

And some images will be written in disk so you can check the original image and the image obtained by the model we just have trained. 


Any doubts or suggestions please open an issue. 
