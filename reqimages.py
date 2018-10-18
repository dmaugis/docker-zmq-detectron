#!/usr/bin/env python
# -*- coding: utf-8 -*-

help = """reqimages
 
Usage:
  reqfiles.py <files>...
 
Options:
  -h --help          This help.
 
(c) Sample Copyright
"""


import zmq
import cv2
import numpy as np
import zmqnparray as zmqa
from docopt import docopt
import os
import os.path
import colorcet as cc
from matplotlib.colors import hex2color


arguments = docopt(help)

import random
lut=[ hex2color(h) for h in cc.palette["fire"]]
random.shuffle(lut)
lut=(256.0*np.array(lut)).astype(np.uint8)


file_list=arguments.pop("<files>", None)

context = zmq.Context()

#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

for fname in file_list:
    if os.path.isfile(fname) and os.access(fname, os.R_OK):
        A=cv2.imread(fname,1)
        if A is None:
            print("[%s] Could not read image" % (fname))
        else:
            arguments['fname']=fname
            print("[%s] Sending request… " % (fname) )
            zmqa.send(socket,A,extra=arguments)
            #zmqa.send(socket,A)
            #  Get the reply.
            B,extra= zmqa.recv(socket)
            print("[%s] Received reply %s" % (fname,str(extra)))
            cv2.imshow('request',A)
            if B is not None:
                s0, s1 = B.shape
                im_color = np.empty(shape=(s0, s1, 3), dtype=np.uint8)
                for i in range(3):
                    im_color[..., i] = cv2.LUT(B, lut[:, i])
                cv2.imshow('reply',im_color)
            cv2.waitKey(0)
    else:
        print("[%s] could not access file" % (fname))


cv2.destroyAllWindows()
