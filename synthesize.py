import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from urllib.request import urlopen
import time
import csv
import sys
import pandas as pd
import json
import requests
import docker


from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.pipeline import StandardOptions

client = docker.from_env()
options = PipelineOptions()
p = beam.Pipeline(options=options)

class JsonCoder(object):
  """A JSON coder interpreting each line as a JSON string."""
  def encode(self, x):
    return json.dumps(x)

  def decode(self, x):
    return json.loads(x)

def compute_data(record):
    client.containers.run("benchmark-app", detach=True)
    url = 'http://127.0.0.1:5000/benchmark'
    response = (requests.post(url, json=record)).text
    res_dic = json.loads(response)
    synthetic_data = res_dic['synthetic']
    del res_dic['synthetic']
    benchmark_pd = pd.DataFrame(res_dic)

    print(synthetic_data)
    print(benchmark_pd)



data_from_source = (p
                    |'Read' >> beam.io.ReadFromText("input.txt",coder= JsonCoder())
					|'Compute Data' >> beam.FlatMap(compute_data))

result = p.run().wait_until_finish()
