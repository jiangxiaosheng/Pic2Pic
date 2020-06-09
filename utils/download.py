import os
import requests


def download(dir, filename, url):
    resource = requests.get(url)
    with open(os.path.join(dir, filename), 'wb') as f:
        f.write(resource.content)
