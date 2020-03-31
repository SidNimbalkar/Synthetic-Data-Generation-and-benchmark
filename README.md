# Synthetic-Data-Generation-and-benchmark

## Overview

1. We use MIT's SDGym framework and generalize it, so it can be used with any random dataset
2. We build an API service which will take in a json input and provide us with details regarding which dataset to injest, and which synthesizer to use. The API service will then use the listed synthesizers and output some synthetic data, the application will also benchmark the synthesizer and output the score using various ML models. We then dockerize the given application.
3. We build an Apache Beam pipeline which will injest multiple datasets at once and perform data synthesis and benchmark parallely, and write the output to GCP bucket. We will use Google Dataflow to execute the pipeline. 
4. We will use streamlit to test the application.



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

### Install Anaconda

Follow the instructions for your operating system.

* For macOS, follow this [link](https://docs.anaconda.com/anaconda/install/mac-os/)
* For Windows, follow this [link](https://docs.anaconda.com/anaconda/install/windows/)


### Install Sublime

Follow the [instructions](https://www.sublimetext.com/3) for your operating system.\
If you already have a prefered text editor, skip this step.


## Run Sequence

1. Run requirements.txt
```
pip install -U -r requirements.txt
```
This command will instal all the required packages and update any older packages.

2. Now that we have our enviornment set up, we will create an S3 bucket.

Follow this [link](https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html) and create two buckets, an input bucket and a outbucket and configure them in the pipeline files.

3. Now, run the pipelines in the following order, you find the instructions of running each pipeline in their respective folders:\
a. Run the Annotation Pipeline `testp.py` to get the labelled dataset\
b. Run the Training Pipeline `training_pipeline.py` to get the trained model\
c. Start the micro service by running the docker `MicroSerivce` from the Microservice folder.\
d. Run the Inference Pipeline `inference_pipeline.py` to get the final output

4. At the end you should get a csv file with sentence and a sentiment, which will look like this:

![alt text](https://github.com/siddhant07/CaseStudy2/blob/master/Images/Final_outout.png)


[CodeLab](https://codelabs-preview.appspot.com/?file_id=1VQEfSxPcW4bHluo56Xh6sZAMK31DOV8JX8PxmrUhIm4#0)
