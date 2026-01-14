docker run --rm -ti -p 8888:8888 --gpus all -v ${PWD}:/root/projects -w /root/projects device-query bash -c "jupyter notebook --ip 0.0.0.0 --no-browser --allow-root &" 
