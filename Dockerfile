FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
# Create app directory
WORKDIR /nlp_tagger

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
WORKDIR /nlp_tagger/src
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader stopwords
RUN python3 train.py
EXPOSE 8000
CMD [ "uvicorn", "api:app", "--reload"]