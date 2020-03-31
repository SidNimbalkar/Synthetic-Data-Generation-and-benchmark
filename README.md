# Synthetic-Data-Generation-and-benchmark

## Overview

1. We use MIT's SDGym framework and generalize it, so it can be used with any random dataset
2. We build an API service which will take in a json input and provide us with details regarding which dataset to injest, and which synthesizer to use. The API service will then use the listed synthesizers and output some synthetic data, the application will also benchmark the synthesizer and output the score using various ML models. We then dockerize the given application.
3. We build an Apache Beam pipeline which will injest multiple datasets at once and perform data synthesis and benchmark parallely, and write the output to GCP bucket. We will use Google Dataflow to execute the pipeline. 
4. We will use streamlit to test the application.




[CodeLab](https://codelabs-preview.appspot.com/?file_id=1VQEfSxPcW4bHluo56Xh6sZAMK31DOV8JX8PxmrUhIm4#0)
