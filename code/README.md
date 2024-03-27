# SmartEnergyOptimizationRL

Create a virtual environment:
```
sudo apt-get install python-virtualenv virtualenv
virtualenv env_sinergym --python=python3.10
source env_sinergym/bin/activate
```


Install [Sinergym](https://github.com/ugr-sail/sinergym):
```
git clone https://github.com/ugr-sail/sinergym.git
cd sinergym
pip install -e .[extras]
```

Install [EnergyPlus 23.1.0](https://github.com/NREL/EnergyPlus/releases/tag/v23.1.0) (the following is for Ubuntu 22.04)
```
wget https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
chmod +x EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
./EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
rm EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
```
I suggest to install EnergyPlus in ./EnergyPlus-23-1-0 (instead of /usr/local/EnergyPlus-23-1-0) and set the simbolic link location to env_sinergym/bin/ just to keep everything local and easy to delete.

Add the EnergyPlus path to the system path and set the EnergyPlus path as an environment variable
```
export PATH=$PATH:./EnergyPlus-23-1-0
export EPLUS_PATH=./EnergyPlus-23-1-0
```