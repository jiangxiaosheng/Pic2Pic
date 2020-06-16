import os

import pandas as pd

records_filename = 'records.csv'
allowed_type = ['jpg', 'png', 'bmp', 'gif']


def make_records(samples_dir):
    with open(records_filename, 'w') as file:
        file.write('img,cls')
        for root, _, files in os.walk(samples_dir, topdown=True):
            cls = root.split('\\')[-1]
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
        self.classes = set(self.data["cls"])

    def get_data(self):
        return self.data

    def get_classes(self):
        return self.classes

    def get_images_by_class(self, classes=None):
        images = []
        for i in range(len(self.data['cls'])):
            cls = self.data['cls'][i]
            if classes is not None and cls not in classes:
                break
            images.append({
                'img': self.data['img'][i],
                'cls': cls
            })
        return images


if __name__ == '__main__':
    samples = Samples('dataset')
    images = samples.get_images_by_class(['accordion'])
    for d in images:
        print(d['img'], d['cls'])
