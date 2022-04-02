# steering_prediction
This is a ROS package to predict the steering from an Embedding.
The Embedding is coming from an Autoencoder between Depth Image and Optical Flow

The model used in this package is coming from the Paper:

> Excavating the Potential Capacity of Self-Supervised Monocular Depth Estimation
>
> Rui Peng, Ronggang Wang, Yawen Lai, Luyang Tang, Yangang Cai
>
> ICCV 2021 ([arxiv](https://arxiv.org/abs/2109.12484))

Please download the model from https://drive.google.com/file/d/1Z60MI_UdTHfoSFSFwLI39yfe8njEN6Kp/view and put the file into src/models/
