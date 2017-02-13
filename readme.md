# Style transfer

Two style transfer methods, based on:
1. [Leon A. Gatys, Alexander S. Ecker, Matthias Bethge](https://arxiv.org/abs/1508.06576)
2. [Justin Johnson, Alexandre Alahi, Li Fei-Fei](https://arxiv.org/abs/1603.08155)

* The implementations are created in torch, and use a pre-trained vgg network, taken via the method described in the [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)


* In order to run an experiment, run:
   * run_experiment_set.py <experiment.csv> [--show-images] [--noterm]
   * --show-images renders the images at each iteration of optimization
   * --noterm runs the experiments in the file in the background, independent of the terminal they were started from.

* Each line in the experiment.csv file should have the following format:
   * train_method,experiment_label,regularize_term1,regularize_term2,regularize_term3,content_image,style_image,results_folder,save_every_count,content_convolution_layer,gpu_number,style_convolution_layers


