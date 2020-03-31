# Synthetic-Data-Generation-and-benchmark

## Overview

1. We use MIT's SDGym framework and generalize it, so it can be used with any random dataset
2. We build an API service (Flask App) which will take in a json input and provide us with details regarding which dataset to injest, and which synthesizer to use. The API service will then use the listed synthesizers and output some synthetic data, the application will also benchmark the synthesizer and output the score using various ML models. We then dockerize the given application.
3. We build an Apache Beam pipeline which will injest multiple datasets at once and perform data synthesis and benchmark parallely, and write the output to GCP bucket. We will use Google Dataflow to execute the pipeline. 
4. We will use streamlit to test the application.

## Design

1. We used Streamlit.io for this step to generate JSON and NPZ files and store them to be sent to the API to generate the outputs.

![alt text](https://github.com/SidNimbalkar/Synthetic-Data-Generation-and-benchmark/blob/master/images/streamlit.png)

2. Now that we have our data, we spin up the Docker image which will start our Application (Flask API). The API will produce Synthetic Data and Benchmark the synthesizer.

![alt text](https://github.com/SidNimbalkar/Synthetic-Data-Generation-and-benchmark/blob/master/images/docker.png)

3. Now we use Apache Beam with Google Dataflow to access the dockerized API by sending it a json file consisting of multiple datasets, so it runs in an embarrassingly parallel way and invokes multiple docker containers.

  - Apache Beam Flow

![alt text](https://github.com/SidNimbalkar/Synthetic-Data-Generation-and-benchmark/blob/master/images/pipeline.png)

  - Google Dataflow visualization 

![alt text](https://github.com/SidNimbalkar/Synthetic-Data-Generation-and-benchmark/blob/master/images/flow.png)

## Installation

### Create an Google Cloud Platform (GCP) account.

If you already have an account, skip this step.

Go to this [link](https://cloud.google.com/gcp/getting-started) and follow the instructions. You will need a valid debit or credit card. You will not be charged, it is only to validate your ID.

### Install Postman

Follow the instructions of your operating system:

[macOS](https://learning.postman.com/docs/postman/launching-postman/installation-and-updates/#installing-postman-on-mac)

[Windows](https://learning.postman.com/docs/postman/launching-postman/installation-and-updates/#installing-postman-on-windows)

### Install Docker

Install Docker Desktop. Use one of the links below to download the proper Docker application depending on your operating system. Create a DockerHub account if asked.

* For macOS, follow this [link](https://docs.docker.com/docker-for-mac/install/).

* For Windows 10 64-bit Home, follow this [link](https://docs.docker.com/docker-for-windows/install/)

 i.  Excecute the files "first.bat" and "second.bat" in order, as administrator.

 ii. Restart your computer.

 iii.Excecute the following commands in terminal, as administrator.
 
     ```
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /f /v EditionID /t REG_SZ /d "Professional"
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /f /v ProductName /t REG_SZ /d "Windows 10 Pro"
     ```
     
 iv. Follow this [link](https://docs.docker.com/docker-for-windows/install/) to install Docker.
 
 v.  Restart your computer, do not log out.

 vi. Excecute the following commands in terminal, as administrator.
 
     ```
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /v EditionID /t REG_SZ /d "Core"\
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /v ProductName /t REG_SZ /d "Windows 10 Home"
     ```

Open a Terminal window and type `docker run hello-world` to make sure Docker is installed properly . It should appear the following message:

`` Hello from Docker!``  
``This message shows that your installation appears to be working correctly.``

Finally, in the Terminal window excecute `docker pull tensorflow/tensorflow:2.1.0-py3-jupyter`.


### Install Sublime

Follow the [instructions](https://www.sublimetext.com/3) for your operating system.\
If you already have a prefered text editor, skip this step.


## Run Sequence

1. Run requirements.txt
```
pip install -U -r requirements.txt
```
This command will instal all the required packages and update any older packages.

2. Now that we have our enviornment set up, we will create an GCP bucket.

Follow this [link](https://cloud.google.com/storage/docs/creating-buckets) and create two buckets, an input bucket and a outbucket and configure them in the pipeline files.

3. Now, run the docker (flask app) using the following instructions,
 a. `docker build -t benchmark-app:latest .` -- this references the `Dockerfile` at `.` (current directory) to build our Docker image & tags the docker image with `benchmark-app:latest`

 b. `docker run -d -p 5000:5000 benchmark-app ` -- Spins up a Flask server that accepts POST requests at http://127.0.0.1:5000/benchmark

4. Now, we run the apache beam pipeline 
    Run the Pipeline using the following command `python synthesize.py` to get the synthetic data and benchmark scores in the GCP bucket

5. At the end you should get a csv file with synthetic data and benchmark scores, which will look like this:

![alt text](https://github.com/SidNimbalkar/Synthetic-Data-Generation-and-benchmark/blob/master/images/bucket.png)

6. We can use our Streamlit application to test out our entire application. Which looks like this:

(upload the streamlit output image in /images folder)

![alt text]('insert link here')

[CodeLab](https://codelabs-preview.appspot.com/?file_id=1VQEfSxPcW4bHluo56Xh6sZAMK31DOV8JX8PxmrUhIm4#0)
