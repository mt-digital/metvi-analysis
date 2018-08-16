# metvi-analysis

To run the analyses you must [download the data](http://mt.digital/static/data/metvi-data.tar.gz) (609 MB).
To follow the following instructions, download the archive to this project directory. When you
unzip the archive,  
MongoDB must be installed and running (see the [Install section of the MongoDB manual](https://docs.mongodb.com/manual/installation/)
for details for your OS).

Unpack the downloaded archive, then use `mongorestore` to load up the metacorps database:
```
cd data && mongorestore viomet-mongodump/
```

Now you are ready to run the analyses. Use `make_viomet_pubdata.py` and `correlate_tweets.py` to generate
the analyses found in the paper. These fetch two .csv files, one for 2012 and one for 2016, that contain all 
annotations we used in the paper. If you'd like to edit our annotations to make your own, you can run
[metacorps](https://github.com/mt-digital/metacorps) locally, then re-run the analyses here 
by modifying `make_viomet_pubdata.py` and `correlate_tweets.py` to use your modified local
database instead of the .csvs. For reference, those .csv files are http://metacorps.io/static/data/viomet-2012-snapshot-project-df.csv and http://metacorps.io/static/data/viomet-2016-snapshot-project-df.csv, respectively.

If you have any trouble, please open an Issue!
