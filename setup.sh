conda config --set channel_priority strict && \
conda create -c conda-forge --name ringer-zero root==6.36.06 python==3.12.12 && \
conda run --live-stream -n ringer-zero pip install -r requirements.txt