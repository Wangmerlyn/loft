source /opt/conda/etc/profile.d/conda.sh 
conda create --name loft python==3.11 -y
conda activate loft
which python
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install absl-py==2.1.0
pip install scipy==1.14.0
pip install wget==3.2
pip install opencv-python==4.10.0.84
pip install tqdm==4.66.4
pip install attrs==24.2.0
pip install pillow==10.4.0
pip install vertexai
pip install transformers==4.42.4
pip install accelerate
pip install flash-attn==2.5.6