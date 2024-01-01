"""Usage: python36 read_data.py some_id from the directory this file is located"""

import tobii_research as tr
import time
import datetime
import ctypes
import json
import pandas as pd
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("person_ID", type=str, help="Person ID")
parser.add_argument("--folder", type=str, default="C:/UnityGame/Game2205.3/Game4_0_Data/DataLog/", help="Folder where eyetracker data is saved")
opt = parser.parse_args()



class GazeData():
    """Class that saves collected data of eyetracker."""
    def __init__(self, person_ID, directory=None):
        self.gaze_data = [] # list object to store collected data.
        self.person_ID = person_ID # ID of list.
        # Directory where save data.
        self.dir = '.' if not directory else directory

        # Create directory in case non existant.
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        self.tracking = False # Set tracking mode
        self.counter = 1 # To count each eye tracking session
        fname = 'eyetrackerdata_' + self.person_ID + '-' + str(self.counter)
        assert fname not in os.listdir(self.dir), 'File already exists'
        # Find eyetracker if class object is associated with person.
        if self.person_ID:
            found_eyetrackers = tr.find_all_eyetrackers()
            self.eyetracker = found_eyetrackers[0]

            # Print characteristics of eyetracker
            print("Address: " + self.eyetracker.address)
            print("Model: " + self.eyetracker.model)
            print("Name (It's OK if this is empty): " 
                  + self.eyetracker.device_name)
            print("Serial number: " + self.eyetracker.serial_number)

    def set_dir(self, folder:str):
        """Set folder where data will be stored.
        
        Params:
            folder (str): absolute path of folder.
        Output:
            void
        """
        self.dir = folder
    
    def collect_data(self):
        """Turn on eye tracking."""
        self.tracking = True
        self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA,
                                     self.__save_gaze_data,
                                     as_dictionary=True)
        while self.tracking:
            time.sleep(5)
            self.save_data()

    def stop_collection(self, save=True):
        """Turn off eye tracker and save collected data."""
        self.eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA,
                                         self.__save_gaze_data)
        if save:
            self.save_data()
        
        # Count each collection of data.
        self.counter += 1
        self.tracking = False
    
    def save_data(self):
        fname = 'eyetrackerdata_' + self.person_ID + '-' + str(self.counter)
        # data = self.converttime(self.gaze_data)
        with open(self.dir + fname, 'w') as fout:
            json.dump(self.gaze_data, fout, indent=2)

    def read_json(self, path:str):
        """Read json file.
        
        This function is useful in case that this class object
        is used to read collected data rather than collect data
        associated with a person
        """
        with open(path, "r") as read_file:
            data = json.load(read_file)
        self.data = pd.DataFrame.from_dict(data)
        return self.data

    def converttime(self, data):
        tmp = [d['system_time_stamp'] for d in data]
        data['Timestamp'] = np.vectorize(self._converttime)(tmp)
        return data

    def _converttime(self, timestamp):
        last_reset = self._last_reset_time()

        microseconds = int(str(timestamp)[-6:])
        seconds = int(str(timestamp)[:-6])

        # extracting hours, minutes, seconds & days from t
        # variable (which stores total time in seconds)
        mins, seconds = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        seconds = hours*3600 + mins*60 + seconds

        delta = datetime.timedelta(days, seconds, microseconds)
        timestamp_c = last_reset + delta

        return timestamp_c

    def _last_reset_time(self):
        # getting the library in which GetTickCount64() resides
        lib = ctypes.windll.kernel32
        
        # calling the function and storing the return value
        t = lib.GetTickCount64()

        # since the time is in milliseconds i.e. 1000 * seconds
        # therefore truncating the value
        microseconds = int(str(t)[-3:] + '000')
        seconds = int(str(t)[:-3])

        # extracting hours, minutes, seconds & days from t
        # variable (which stores total time in seconds)
        mins, seconds = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)

        seconds = hours*3600 + mins*60 + seconds

        delta = datetime.timedelta(days, seconds, microseconds)
        timestamp = time.time()

        value = datetime.datetime.fromtimestamp(timestamp)
        reset_day = value - delta
        return reset_day

    def __save_gaze_data(self, gaze_data):
        print("Left eye: ({gaze_left_eye}) \t Right eye: ({gaze_right_eye})".format(
        gaze_left_eye=gaze_data['left_gaze_point_on_display_area'],
        gaze_right_eye=gaze_data['right_gaze_point_on_display_area']))
        now = time.time()
        gaze_data['Timestamp'] = str(datetime.datetime.fromtimestamp(now))

        # Add AOI

        self.gaze_data.append(gaze_data)

    def clean(self, start, end):
        """Remove data before start and after end time.
        
        Params:
            start (Timestamp)
            end (Timestamp)
        
        """
        condition  = (self.data.Timestamp > start) & (self.data.Timestamp < end)
        self.data = self.data[condition]


if __name__=='__main__':
    directory = opt.folder + opt.person_ID + '/'
    gd = GazeData(opt.person_ID, directory)

    try:
        gd.collect_data()
    except KeyboardInterrupt:
        gd.stop_collection()
