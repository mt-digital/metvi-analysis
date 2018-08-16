# metvi-analysis

To run the analyses you must [download the data](http://mt.digital/static/data/metvi-data.tar.gz).
To follow the following instructions, download the archive to this project directory. When you
unzip the archive,  
MongoDB must be installed and running (see the [Install section of the MongoDB manual](https://docs.mongodb.com/manual/installation/)
for details for your OS).

Unpack the downloaded archive, then use `mongorestore` to load up the metacorps database:
```
cd data && mongorestore viomet-mongodump/
```

Now you are ready to run the analyses. Use `make_viomet_pubdata.py` and `correlate_tweets.py` to generate
the analyses found in the paper.
