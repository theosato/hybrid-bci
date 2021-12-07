# set base image (host OS)
FROM python:3.9

# copy app content to src container folder
COPY . /src

# set the working directory in the container
WORKDIR /src

# install dependencies
RUN pip install -r requirements.txt

# command to run on container start
CMD [ "python", "app.py" ]