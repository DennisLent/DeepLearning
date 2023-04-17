# Deep Learning
setup virtual environment (python 3.10 or higher)

python3 -m venv --system-site-packages deep_env
source deep_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt



to do:
- write own convolutional network using numpy/scipy/sklearn
- update network to improve accuracy
- v2 (->conv32, pooling2,2, conv32, pooling2,2, conv64, flatten, dense10) with Test accuracy = 0.5235002040863037 (10 epochs)
- v3 (->conv32, pooling2,2, conv32, pooling2,2, conv64, flatten, dense64, dense10) with Test accuracy = 0.3882308006286621 (10 epochs)
- v3 (->conv64, pooling2,2, conv64, pooling2,2, conv128, flatten, dense10) with Test accuracy = 0.5475735664367676 (10 epochs)
- v4 (->conv128, pooling2,2, conv64, pooling2,2, conv64, flatten, dense10) with Test accuracy = 0.5735574960708618 (10 epochs)
