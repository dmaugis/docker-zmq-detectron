# Use facebookresearch/detectron as parent image
FROM facebookresearch/detectron

# add model to docker
ADD model_final.pkl model_final.pkl

# install my 'standard' set of development tools
COPY install-tools.sh /tmp/
RUN  /tmp/install-tools.sh  
RUN  rm /tmp/install-tools.sh

COPY install-python.sh /tmp/
RUN  /tmp/install-python.sh  
RUN  rm /tmp/install-python.sh

# install install-zmq4.2.5.sh
COPY install-zmq4.2.5.sh /tmp/
RUN  /tmp/install-zmq4.2.5.sh  
RUN  rm /tmp/install-zmq4.2.5.sh

# add custom script
ADD zmqnparray.py    /detectron/tools/
ADD zmqrep_detectron.py /detectron/tools/

CMD bash



