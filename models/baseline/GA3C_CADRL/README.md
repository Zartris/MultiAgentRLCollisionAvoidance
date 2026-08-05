# GA3C CADRL
This one is a bit annoying as it relies on tensorflow 1.5.0. which is an old version of tensorflow. So if you want to make this work, you need to install python3.6 and tensorflow 1.5.0. but the simulator does not support these old versions of all the libraries.
so to make this work, we make a flask server that runs the GA3C model and the simulator runs on a different server. The simulator sends the data to the GA3C server and the GA3C server sends the action back to the simulator.

## How to run
1. Install the dependencies
```bash
sudo apt-get install python3.6 python3-pip
pip3 install flask numpy matplotlib tensorflow==1.5.0
```
2. Run the server
```bash
python server.py
```
3. Run the simulator
```bash
python simulator.py
```
