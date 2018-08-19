#!/bin/bash

name=trt_gen_tool
image=dev/tensorrt
docker rm -f ${name}

params=$@
nvidia-docker run -ti \
  --name ${name} \
  -v `pwd`/../../:/work \
  ${image} \
  /bin/bash -c "cd /work/app/trt_gen_tool && ./run_locally.sh ${params}"
