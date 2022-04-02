#!/usr/bin/env python3
import sys
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
from imantics import Mask
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
import rospkg
from dynamic_reconfigure import client
import time

from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Action_Predicter_Models import Action_Predicter_Dense

from PIL import Image as ImagePIL

use_gpu = True


SUBSCRIBING_TOPIC_NAME = 'opt_flow/embedding'


PUBLISHING_TOPIC_NAME = 'cmd_vel'


class Steering_Prediction_Service:

    def __init__(self):

        rospack = rospkg.RosPack()
        base_path = rospack.get_path("steering_prediction")

        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

        self.model = Action_Predicter_Dense()
        self.model.load_state_dict(torch.load(base_path+"/src/models/Action_Predicter_Same.pt",map_location=torch.device(self.device)))


        print('model_loaded')

        self.publisher = rospy.Publisher(PUBLISHING_TOPIC_NAME, Twist, queue_size=1)




    def predict(self, embedding):

        emb_tensor = torch.tensor(embedding)#.copy())

        steering = self.model(emb_tensor)

        return steering




    def steering_predict(self, embedding):

        output = self.predict(embedding.data)

        msg = Twist()
        msg.linear.x = 1.0
        msg.angular.z = output[0].cpu().detach().numpy()

        self.publisher.publish(msg)


def steering_prediction():
    rospy.init_node("Steering_Prediction", anonymous=True)

    rospy.loginfo('Start predicting cmd_vel from Embedding')
    steering_node = Steering_Prediction_Service()
    rospy.Subscriber(SUBSCRIBING_TOPIC_NAME, Float32MultiArray, steering_node.steering_predict,queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        print("Node started")
        steering_prediction()
    except rospy.ROSInterruptException:
        pass
