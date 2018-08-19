To download TensorRT tarball

```
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/4.0/ga/TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-8.0.cudnn7.1
mv TensorRT-4.0.1.6.Ubuntu-16.04.4.x86_64-gnu.cuda-8.0.cudnn7.1.tar.gz ${your_repo}/dockerfiles/tarball/nvidia/
```
To download gflags tarball

```
wget https://github.com/gflags/gflags/archive/v2.2.0.tar.gz
mv gflags-2.2.0.tar.gz ${your_repo}/dockerfiles/tarball/google/
```
To build docker image

```
cd ${your_repo}/dockerfiles && \
docker build -t {your_docker_image_name}
```