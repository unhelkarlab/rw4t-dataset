Set of scripts to collect and process physiological and behavioral data from Unity engine

# Unity Data
1. Download the RW4T Simulator. The data will be collected automatically.
2. Get discrete and continuous data in the (inverse) reinforcement learning paradigm by running `python extract.py`

# BioHarness Data
### Prerequisites
1. Clone [App-Zephyr](https://github.com/labstreaminglayer/App-Zephyr) and follow the instructions in its README.
2. Install [pylsl](https://pypi.org/project/pylsl/), which is the Python interface to the Lab Streaming Layer (LSL).

### Collecting Data
1. Turn on BioHarness and check its LED lights to make sure it has the expected status.
2. Open a Miniconda terminal and invoke the scirpt ```run.cmd``` in App-Zephyr to connect BioHarness to LSL.
3. Open a regular terminal and invoke the script ```collect.py``` in this bioharness directory and pass in appropriate arguments to stream BioHarness data and store them locally.

# Tobii Pro EyeTracker Data
1. Turn on and calibrate EyeTracker. We used the [Tobii Pro eye tracker manager](https://www.tobii.com/products/software/applications-and-developer-kits/tobii-pro-eye-tracker-manager)
2. Run script ```python collect_eyetracker.py```
3. To stop collection and save, press Ctrl+C once.
