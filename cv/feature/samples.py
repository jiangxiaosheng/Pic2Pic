import os

import pandas as pd

records_filename = 'records.csv'
allowed_type = ['jpg', 'png', 'bmp', 'gif']


def make_records(samples_dir):
    with open(records_filename, 'w') as file:
        file.write('image,class')
        for root, _, files in os.walk(samples_dir, topdown=True):
            cls = root.split('/')[-1]
            for name in files:
                suffix = name.split('.')[-1]
                if suffix not in allowed_type:
                    continue
                img = os.path.join(root, name)
                file.write("\n{},{}".format(img, cls))


class Samples:
    def __init__(self, samples_dir):
        make_records(samples_dir)
        self.data = pd.read_csv(records_filename)
        self.classes = set(self.data["class"])

    def get_data(self):
        return self.data
