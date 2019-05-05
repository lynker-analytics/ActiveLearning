All examples run on 
ubuntu 18.04
Python 3.6.7
Tensorflow 1.12.0-rc0
Keras 2.2.4

Also used: scikit-learn and pandas

Also, for the web page, I used sql server developer edition (for linux) as the database backend and php 7 for the front end with and the odbc driver {ODBC Driver 17 for SQL Server} to talk between the two.

#presentations

in the presentations folder you'll find two versions of the slide show.
"Meetup_ActiveLearning_May2019.pptx" is the warts and all power point as presented at the May 2019 meetup.
"Meetup_ActiveLearning_May2019_NOVIDEO.pptx" is the same minus the video slide that shows the speeded up webform interaction. The original is a large file. This is smaller.


https://github.com/lynker-analytics/ActiveLearning/presentation/webformtimelapse.mp4

#Scripts looking at CIFAR 10
##  Samples Sizes
### the following will produce csv output that shows the test and validation accuracacy when training a resnet20 model on a subset of the cifar10 training data
### this script is used to demonstrate the effect of training set size on model accuracy against the test set.
```
python3 ./sample_sizes.py > logs/output.samples_sizes.log
cat logs/output.samples_sizes.log
```

##  Active Learning on CIFAR 10
### the following script collects data to compare different active learning sampling methods. Note, the "NULL" method is a random sampling and so should be the same as "passive" learning and gives us baseline performance.
### the output should be comman separated data that can be pasted into excel or similar
```
python3 ./Active_Sampling_CIFAR10.py > logs/output.Active_Sampling_CIFAR10.log
cat logs/output.Active_Sampling_CIFAR10.log
```

##  Active Learning on CIFAR 10 with single class classification
### the following script collects data to compare active versus passive learning on single class classification using the cifar10 data set
```
python3 ./Active_Sampling_CIFAR10_choose1.py > logs/output.Active_Sampling_CIFAR10_choose1.log
cat logs/output.Active_Sampling_CIFAR10_choose1.log
```

# Active Learning example with web form, human in the loop and asynchronous machine learning bit
## see the web/boris/ subfolder
## to get this runnnig, you'll need to have
### a web server, php 7, sql server, odbc drivers for sql, python and all the above ml libraries working
### on your database create the tables from the *.sql scripts
### set up a user on your database that can be used by the odbc connection from php
### update the connection details in the boris.php file
### update the database connection details in the web/boris/ml/config.py file
###
### put your images into the unclassified/ folder
###
### holdout images
### some preclassified known images will need to be in the boris_classify table with a createdby column = 'holdout'.  This is the data that is used to calculate the validation_score.
###
### seed data
### as you start classifying images through the webform, don't kick off the machine learning part until you have some samples classified already. I classified 100 images (approx) before kicking off the machine learning script.
###
###
## The machine learning part:
### see web/boris/ml/
### the run.sh script will loop between a training script and an inference script and will update the database with some inference results. These samples with proposed labels will then be served up by the webform - all going well! 
### in a bash shell or similar run the following to kick off the machine learning processes
```
sh ./run.sh
```
