# kNN, Naive Bayes and kMeans'
A python program demonstrating three supervised learning algorithms: kNN, Naive Bayes and kMeans
Compute Precision and Recall for each label in the test data

## File structure -  
1. `learn.py` - main file 


### Observations about the code -
- Use `python3` for running the code
- Both the test and training file will be a csv file where each line represents one 'record'.  There will be no header line.
- It is illegal for both K and C to be greater than 0 (they both default to 0)


## Sample command:
> kNN
`python learn.py -train knn3.train.txt -test knn3.test.txt -K 7 -v`

> Naive Bayes
`python learn.py -train ex2_train.csv -test ex2_test.csv -C 2 -v`

> kMeans
`python learn.py -train km2.txt 0,0,0 200,200,200 500,500,500 -d manh`

where:

`-v` : indicates to print verbose output. If present, it prints each predicted vs actual label for kNN and Naive Bayes; type: optional, string

`-train` : the training file for kNN, Naive Bayes, kMeans; type: required, string

`-test` : the testing file for kNN and Naive Bayes; type: required, string

`-K` : if > 0 indicates to use kNN and also the value of K (if 0, do Naive Bayes')

`-C` : if > 0 indicates the Laplacian correction to use (0 means don't use one)

`-d e2` or `-d manh` :  indicating euclidean distance squared or manhattan distance to use for kMeans

arguments: if a list of centroids is provided those should be used for kMeans


## Requirements
python3
numpy
pandas