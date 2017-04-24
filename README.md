#**Introduction**

This the second project for the CSE353 Machine Learning course at Stony Brook University.

#**Installation**

You need Python 3.5 to run this project. For instruction on downloading and installing Python, see [here](https://www.python.org/downloads). sklearn and numpy are also required, the easiest way to install them is to ues pip. If you already have Python installed, you can run the following command to install the latest release of SciPy:
```commandline
pip install SciPy
```

and SciKit learn:
```commandline
pip install scikit-learn
```

#**Usage**
To run the preprocessing function alone, you can do that from either the command line or the notebook. 

To run the function from command line, go to the dir where the python file is located, the type the following:
```commandline
python preprocessing.py
```
the options for this are punctuation and stopwords
to keep punc and stopwords use:
```commandline
python preprocessing.py --punc --stop
```

to run k nearest neighbor:
```commandline
python runner.py
```
there are two more options, --distance and --k for distance metrics and k value respectively