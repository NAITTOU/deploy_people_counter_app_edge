#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        
    def load_model(self ,model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.network = IENetwork(model=model_xml, weights=model_bin)
      
        self.plugin = IECore()
        ### TODO: Check for supported layers ###
        
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
        
        ### TODO: Add any necessary extensions ###
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers) != 0:
            
            log.warning("Unsupported layers found: {}".format(unsupported_layers))
            
            if cpu_extension and "CPU" in device:
                
                log.info("Adding a CPU extension ...")
                self.plugin.add_extension(cpu_extension, device)
                log.info("The CPU extension was added")
                
                supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                
                if len(unsupported_layers) != 0:
                    
                    log.warning("There Still Unsuported layers even after the extension was added {}".format(unsupported_layers))
                    log.info("exiting program ...")
                    exit(1)
                
                log.info("All the layers of you model are supported now by the Inference engine ")
                     
            else:
                
                log.ERROR("Check whether extensions are available to add to IECore.")
                log.info("exiting program ...")
                exit(1)
         
        
        self.net_plugin = self.plugin.load_network(self.network, device)
        self.input_blob = sorted(self.network.inputs)
        self.output_blob = next(iter(self.network.outputs))
        
        ### TODO: Return the loaded inference plugin ###
  
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        
        return  [self.network.inputs[x].shape for x in  self.input_blob]

    def exec_net(self,image):
        ### TODO: Start an asynchronous request ###
        
        self.infer_request_handle = self.net_plugin.start_async(request_id=0, inputs=image)
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        
        status = self.infer_request_handle.wait(-1)
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        
        output = self.infer_request_handle.outputs[self.output_blob]
        
        ### Note: You may need to update the function parameters. ###
        return output