
import zmq
import numpy as np


def send(socket, array, flags=0, copy=True, track=False,extra=None):
    A = np.ascontiguousarray(array)
    md = {}
    if extra is not None:
       md["extra"]=extra
    if A is not None:
       md['dtype'] = str(A.dtype)
       md['shape'] = A.shape
    socket.send_json(md, flags|zmq.SNDMORE)
    #print "md: ",md
    return socket.send(A, flags, copy=copy, track=track)

def recv(socket, flags=0, copy=True, track=False):
    extra=None
    md = socket.recv_json(flags=flags)
    #print "md: ",md
    extra=md.pop('extra', None)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A=None
    if 'dtype' in md:
       if 'shape' in md:
          if md['dtype']!='object':
             A = np.frombuffer(msg, dtype=md['dtype'])
             A = A.reshape(md['shape'])
    return A, extra


def recvn(socket, flags=0, copy=True, track=False):
    result=[]
    while True:          
        extra=None
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        extra=md.pop('extra', None)
        A = np.frombuffer(msg, dtype=md['dtype'])
        A.reshape(md['shape'])
        result.Append({ A, extra })
        if not self.socket.get(zmq.RCVMORE):
           return result

def sendn(socket, arrays, flags=0, copy=True, track=False,extras=None):
    count=0
    last =len(arrays)
    for array,extra in zip(arrays,extras):
        A = np.ascontiguousarray(array)
        md = dict(
         dtype = str(A.dtype),
         shape = A.shape,
        )
        if extra is not None:
           md["extra"]=extra
        count=count+1
        if count!=last:
           socket.send_json(md, flags|zmq.SNDMORE)
           socket.send(A, flags|zmq.SNDMORE, copy=copy, track=track)
        else:
           socket.send_json(md, flags|zmq.SNDMORE)
           socket.send(A, flags, copy=copy, track=track)
           return True

