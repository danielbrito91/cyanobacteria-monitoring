from urllib import response

import requests
import yaml


def test_sisagua_url():

    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    response = requests.get(config["data_create"]["url"])

    assert response.status_code == 200

def test_s2a():
    pass