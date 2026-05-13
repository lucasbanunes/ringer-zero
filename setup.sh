conda config --set channel_priority strict && \
conda create -c conda-forge --name neuralnet root==6.36.06 python==3.12.12 && \
conda run --live-stream -n neuralnet pip install -r requirements.txt
