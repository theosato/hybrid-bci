# HyBCI - Hybrid BCI with IBK and ErrP paradigms using ConvNets

## Getting Started in 10 Minutes

- Clone this repo 
- Install requirements
- Run the script
- Go to http://localhost:5000
- Done! :tada:

## Run with Docker

With **[Docker](https://www.docker.com)**, you can quickly build and run the entire application in minutes :whale:

```shell
# 1. First, clone the repo
$ git clone https://github.com/theosato/hybrid-bci.git
$ cd hybrid-bci

# 2. Build Docker image
$ docker build --no-cache -t hybci_flask_app .

# 3. Run!
$ docker run -it --rm -p 5000:5000 hybci_flask_app
```

Open http://localhost:5000 and wait till the webpage is loaded.

## Local Installation

It's easy to install and run it on your computer.

```shell
# 1. First, clone the repo
$ git clone https://github.com/theosato/hybrid-bci.git
$ cd hybrid-bci

# 2. Install Python packages
$ pip install -r requirements.txt

# 3. Run!
$ python app.py
```

Open http://localhost:5000 and have fun. :smiley:

------------------
