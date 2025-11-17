Content:
Gymnasium v1.2.2

Install:
python -m venv .venv
source .venv/bin/activate
cd group_project/Gymnasium
pip install -e .

pip install "gymnasium[classic_control]"
pip install matplotlib

CHECK:
% pip list        
Package              Version Editable project location
-------------------- ------- -------------------------------------------------
cloudpickle          3.1.2
Farama-Notifications 0.0.4
gymnasium            1.2.2   ./group_project/Gymnasium
numpy                2.3.5
pip                  24.3.1
typing_extensions    4.15.0


RUN:

Part1:
python mountain_car.py --train --episodes 5000
python mountain_car.py --render --episodes 10

Part2:
python frozen_lake.py
