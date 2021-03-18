FROM python:3.8-slim-buster
# Create app directory
WORKDIR /nlp_tagger

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
WORKDIR /nlp_tagger/src
CMD [ "python3", "train.py" ]
CMD [ "unicorn","main:app","--reload"]