# prepare dependency
sudo apt install -y git python3 python3-pip jupyter
# install the python dependencies
pip3 install jupyter syft torch torchvision pandas
# due to syft compatibility issue, we need to downgrade torch
pip3 install --upgrade torch==1.1.0
# clone the repository
git clone https://github.com/Souhail-MEFTAH/i-dash2019.git
# launch the project
cd i-dash2019/
jupyter notebook
