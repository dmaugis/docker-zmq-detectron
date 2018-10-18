
#nvidia-docker run --rm -it --network host \
#                  -v $(realpath ./results):/detectron/results \
#                  -v $(realpath ./datasets):/detectron/datasets \
#                  facebookresearch/detectron bash

#nvidia-docker run --rm -it --network host \
#                  -v $(realpath ./results):/detectron/results \
#                  -v $(realpath ./datasets):/detectron/datasets \
#                  dmaugis/detectron1 tools/infer_mask.py datasets

nvidia-docker run --rm -it --network host \
                  -v $(realpath ./results):/detectron/results \
                  -v $(realpath ./datasets):/detectron/datasets \
                  dmaugis/detectron1 tools/zmqrep_detectron.py /tmp
