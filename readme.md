### For training :
copy training_ground-truth and training_input to input
go to code folder and run : python baseline_aug.py -d 0.1 -a ReLU > logs_1.txt

### For inference :
copy test_input to input
go to code folder and run : python baseline_aug_predict.py
output images should be in output/baseline_unet_aug_do_0.1_activation_ReLU_
