"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

import numpy as np

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

log.getLogger().setLevel(log.INFO)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_boxes(frame, result, prob_threshold, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    counter = 0
    
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        Class_id = box[1]
        
        if conf >= prob_threshold and Class_id == 1:
            counter +=1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    return counter,frame


def connect_mqtt():
    
    ### TODO: Connect to the MQTT client ###
    
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client

def infer_on_stream(args, client):
    
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    
    infer_network = Network()
    
    # Set Probability threshold for detections
    
    prob_threshold = args.prob_threshold
    model = args.model
    device = args.device
    input_stream = 0 if args.input.upper() == 'CAM' else args.input
    single_image_mode = False
    if input_stream.endswith('.jpg') or input_stream.endswith('.bmp'):
        single_image_mode = True
        log.info("Input is a single Image")
        
    elif input_stream.endswith('.mp4') or input_stream == 0:
        log.info("Input type : Streaming ")
    else:
        
        log.info("Unspported Input !")
        exit(1)
        
    cpu_extension = args.cpu_extension

    ### TODO: Load the model through `infer_network` ###
    
    exec_net = infer_network.load_model(model, device, cpu_extension)
    image_info, image_tensor = infer_network.get_input_shape()
    b, c, i_height, i_width = image_tensor

    ### TODO: Handle the input stream ###
    
    capture  = cv2.VideoCapture(input_stream)
    capture.open(input_stream)
    width = int(capture.get(3))
    height = int(capture.get(4))
    total_count = 0
    prev_count =0
    frames_counter = 0
    last_seen = 0
    last_seen_count = 0
    time_out = 0
    count_to_send = 0


    ### TODO: Loop until stream is over ###
    
    while capture.isOpened():
        
        ### TODO: Read from the video capture ###
        
        flag, frame = capture.read()
        frames_counter += 1
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        if key_pressed == 27:
            break

        ### TODO: Pre-process the image as needed ###
        
        p_frame = cv2.resize(frame, (i_width, i_height))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
                

        ### TODO: Start asynchronous inference for specified request ###
        
        inputs = {'image_tensor' : p_frame,'image_info': (i_height, i_width, 1) }
        
        start_inf = time.time()
        infer_network.exec_net(inputs)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            end_inf = time.time()
            inf_time = end_inf - start_inf
            inf_time = "Inference time: {:.3f}ms".format(inf_time * 1000)
            
            ### TODO: Get the results of the inference request ###
            
            results  = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            
            current_counter ,frame = draw_boxes(frame, results, prob_threshold, width, height)
            cv2.putText(frame,inf_time,(15,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(200,10,10),1)
            
            log.info("Time at : {0:.2f} seconds , current Person count {1} , Inference time : {2} ".format(frames_counter/10,current_counter,inf_time))

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            span = frames_counter - last_seen
            if (current_counter - prev_count) > 0 and span >3:
                
                time_in = frames_counter/10
                total_count += (current_counter - prev_count)
                count_to_send = current_counter
                
                client.publish("person", json.dumps({"total": total_count}))
                client.publish("person/duration", json.dumps({"duration": time_out}))
                  
                log.info("A New Person Entred at  : {0} , Total Person count : {1} , current Person count {2} ,Previous Person count : {3} ,Diffrence : {4}".format(frames_counter/10,total_count,current_counter,prev_count,span))
            
            elif (current_counter - last_seen_count) < 0 and span == 3:
                
                time_out = round((frames_counter/10 - time_in),2)
                count_to_send = (current_counter - last_seen_count)+1
                
                log.info("A Person Left at : {0} , Total Person count : {1} , current Person count {2} ,Previous Person count : {3} ,Diffrence : {5} , Time Spent : {5}".format(frames_counter/10,total_count,current_counter,prev_count,span,time_out))
            
        
        
        client.publish("person", json.dumps({"count": count_to_send}))
        
        ### TODO: Send the frame to the FFMPEG server ###

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        
        
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        
        if current_counter > 0:
            last_seen = frames_counter
            last_seen_count = current_counter
        
        prev_count = current_counter

    #log.info("Total Detected persons Count {0} % , Total Inference time {1} ms ".format((total_count_inf/frames_counter)*100,round(total_inf_time/frames_counter,2)))
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()