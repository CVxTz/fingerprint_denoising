Code for : https://arxiv.org/abs/1807.11888


Medium post : https://towardsdatascience.com/fingerprint-denoising-and-inpainting-using-fully-convolutional-networks-e24714c3233

### For training :
copy training_ground-truth, validation_ground-truth, validation_input and training_input to input ( from https://competitions.codalab.org/competitions/18426 )

go to code folder and run : python baseline_aug.py -d 0.1 -a ReLU > logs_1.txt

### For inference :
copy test_input to input

go to code folder and run : python baseline_aug_predict.py

output images should be in output/baseline_unet_aug_do_0.1_activation_ReLU_
