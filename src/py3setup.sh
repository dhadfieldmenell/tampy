# Get python code
python3 -m pip install numpy seaborn numba pandas pybullet dm_control numdifftools ipdb
python3 -m pip install tensorflow==1.10.0
python3 -m pip install -e BaxterGym
python3 -m pip install h5py psutil
python3 -m pip install --upgrade numpy

cd /opt/gurobi702/linux64

source ~/.bashrc
python3 setup.py install

