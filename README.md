# SmartEnergyOptimizationRL

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
```