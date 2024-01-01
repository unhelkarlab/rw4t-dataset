import csv
import datetime
import argparse
import os

import pylsl
from pylsl import StreamInlet, resolve_stream


parser = argparse.ArgumentParser()
parser.add_argument("--person_ID", type=str, help="Person ID")
parser.add_argument("--folder", type=str, help="Folder where zephyr data is saved")
opt = parser.parse_args()

class BioContainer():
    def __init__(self, person_ID, directory=None):
        self.person_ID = person_ID
        # Directory where save data.
        self.dir = '.' if not directory else directory
        # Create directory in case non existant.
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        fname = 'zephyrdata_' + self.person_ID + '.csv'
        # avoid overwriting
        assert fname not in os.listdir(self.dir), 'File already exists'

        self.last_reset = self._last_reset_time()

    def get_labels(self, info):
        labels = ["Timestamp"]
        channels = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
                    labels.append(channels.child_value("label"))
                    channels = channels.next_sibling()
        return labels
    
    def create_csv(self, info):
        fname = 'zephyrdata_' + self.person_ID + '.csv'
        self.labels = self.get_labels(info)
        with open(self.dir + fname, 'w', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow(self.labels)

    def append_list_as_row(self, timestamp, sample, last_reset):
        timestamp = self._converttime(timestamp, last_reset)
        list_of_elem = [timestamp] + sample
        fname = 'zephyrdata_' + self.person_ID + '.csv'
        # Open file in append mode
        with open(self.dir + fname, 'a+', newline='') as f:
            # Create a writer object from csv module
            csv_writer = csv.writer(f)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
        return timestamp

    def _converttime(self, timestamp, last_reset):
        delta = datetime.timedelta(seconds=timestamp)
        return last_reset + delta

    def _last_reset_time(self):
        """Last reset time using LSL functions"""
        # getting the library in which GetTickCount64() resides
        seconds_transcurred = pylsl.local_clock()
        now = datetime.datetime.now()
        delta = datetime.timedelta(seconds=seconds_transcurred)
        reset_day = now-delta
        return reset_day


def main():
    # first resolve a bioharness stream on the lab network
    print("looking for a bioharness stream...")
    streams = resolve_stream('name', 'ZephyrSummary')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    directory = opt.folder + opt.person_ID + '/'
    dc = BioContainer(opt.person_ID, directory)
    last_reset = dc._last_reset_time()
    dc.create_csv(info)

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        timestamp = dc.append_list_as_row(timestamp, sample, last_reset)
        print(timestamp, "HR", sample[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Finished collecting bioharness data")