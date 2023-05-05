import sys
import numpy as np
import pandas as pd
import logging as log
import warnings
warnings.simplefilter("ignore")

log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)

def read_data(filename):
    """
    Reads data from file and returns a pandas dataframe

    Parameters
    ----------
    filename : str
        Name of the file to be read
    
    Returns
    -------
    df : pandas dataframe
        Dataframe containing the data from the file
    """
    try:
        try:
            if filename.split('.')[-1] == 'txt':
                df = pd.read_csv(filename, sep=" ", header=None)[0].str.split(",", expand = True)
            if filename.split('.')[-1] == 'csv':
                df = pd.read_csv(filename, header=None)
        except:
            log.error(f"Invalid file extension: {filename}")
            exit()
        predictive_attributes = df.columns.values[:-1].tolist()
        predictive_attributes = [f'A{i}' for i in predictive_attributes]
        classification_attribute = df.columns.values[-1].tolist()
        classification_attribute = f'C' 
        df.columns = predictive_attributes + [classification_attribute]
        return df
    except:
        log.error(f"Invalid filename: {filename}")
        exit()


def decide_label(top_k, labels):
    """
    Decides the label for a given test point based on the top k nearest neighbours

    Parameters
    ----------
    top_k : list
        List of tuples containing the distance and label of the top k nearest neighbours
    labels : list
        List of all possible labels

    Returns
    -------
    label : str
        Label of the test point
    """
    # returns label with maximum weight, label in sorted order in case of ties 
    weighted_top_k = [(1.0/i[0], i[1]) for i in top_k]
    decision = {k:0.0 for k in labels}
    for point in weighted_top_k:
        decision[point[1]] += point[0]
    return max(decision, key=decision.get)


def euclidean_distance(pt1, pt2):
    """
    Calculates the euclidean distance between two points
    """
    point1 = np.array(pt1)
    point2 = np.array(pt2)
    return np.sqrt(np.sum(np.square(point1 - point2)))


def manhattan_distance(pt1, pt2):
    """
    Calculates the manhattan distance between two points
    """
    point1 = np.array(pt1)
    point2 = np.array(pt2)
    return np.sum(np.abs(point1 - point2))


def print_metrics(metrics):
    """
    Prints the metrics for a given algorithm

    Parameters
    ----------
    metrics : dict
        Dictionary containing the metrics for each label
    
    Returns
    -------
    None
    """
    for label in metrics.keys():
        print(f"Label={label} Precision={metrics[label]['tp']}/{metrics[label]['tp'] + metrics[label]['fp']} Recall={metrics[label]['tp']}/{metrics[label]['tp'] + metrics[label]['fn']}")


def kNN(train_df, test_df, kNN_k, verbose):
    """
    Performs kNN classification on the given data

    Parameters
    ----------
    train_df : pandas dataframe
        Dataframe containing the training data
    test_df : pandas dataframe
        Dataframe containing the test data
    kNN_k : int
        Number of nearest neighbours to consider
    verbose : bool
        Flag to print verbose output
    
    Returns
    -------
    metrics : dict
        Dictionary containing the metrics for each label
    """
    labels = sorted(train_df.C.unique().tolist())

    metrics = {}
    for v in labels:
        metrics[v] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for _, row in test_df.iterrows():
        test_point = tuple([int(p) for p in row[:-1].values])
        true_test_label = row[-1]
        distance_list = []
        train_label_list = []
        
        for __, train_row in train_df.iterrows():
            train_point = tuple([int(p) for p in train_row[:-1].values])
            distance_list.append(euclidean_distance(train_point, test_point))
            train_label_list.append(train_row[-1])
        top_k = sorted(zip(distance_list, train_label_list))[:kNN_k]
        predicted_test_label = decide_label(top_k=top_k, labels=labels)
        
        if verbose:
            print(f"want={true_test_label} got={predicted_test_label}")
        
        if true_test_label == predicted_test_label:
            metrics[true_test_label]['tp'] += 1
        else:
            metrics[true_test_label]['fn'] += 1
            metrics[predicted_test_label]['fp'] += 1
    
    return metrics  
    

def naive_bayes(train_df, test_df, C_flag, delta, verbose):
    """
    Performs Naive Bayes classification on the given data

    Parameters
    ----------
    train_df : pandas dataframe
        Dataframe containing the training data
    test_df : pandas dataframe
        Dataframe containing the test data
    C_flag : bool
        Flag to indicate whether to use Laplace smoothing
    delta : int
        Value of delta to be used for Laplace smoothing
    verbose : bool
        Flag to print verbose output

    Returns
    -------
    metrics : dict
        Dictionary containing the metrics for each label
    """
    num_train_examples = len(train_df.index)
    labels = sorted(train_df.C.unique().tolist())

    metrics = {}
    for v in labels:
        metrics[v] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for idx, row in test_df.iterrows():
        prob_a_label_list = []
        pred_attribute_values = row.values[:-1]
        for v in labels:
            denominator = len(train_df[train_df['C'] == v].index)
            if verbose:
                print(f"P(C={v}) = [{denominator} / {num_train_examples}]")
            prob_c = denominator/num_train_examples
            prob_a_list = []
            for idx, u in enumerate(pred_attribute_values):
                #  q=#(Dom(Ai)), the number of different values of Ai
                q = len(train_df[f'A{idx}'].unique())
                num = len(train_df[(train_df[f'A{idx}'] == u) & (train_df[f'C'] == v)].index)
                den = denominator
                if C_flag:
                    num += delta
                    den += q*delta    
                prob_a = float(num/den)
                if verbose:
                    print(f"P(A{idx}={u} | C={v}) = {num} / {den}")
                
                prob_a_list.append(prob_a)
            prob_a_label_list.append(np.prod(prob_a_list)*prob_c)   

        if verbose:
            for idx, v in enumerate(sorted(labels)):
                print(f"NB(C={v}) = {prob_a_label_list[idx]:06f}")
        
        predicted_test_label = labels[np.argmax(prob_a_label_list)]
        true_test_label = row.values[-1]
        
        if predicted_test_label == true_test_label:
            metrics[true_test_label]['tp'] += 1
            if verbose:
                print(f'match: "{predicted_test_label}"')
        else:
            metrics[true_test_label]['fn'] += 1
            metrics[predicted_test_label]['fp'] += 1
            if verbose:
                print(f'fail: got "{predicted_test_label}" != want "{true_test_label}"')
    
    return metrics    


def kMeans(train_df, centroids, distance_metric):
    """
    Performs kMeans clustering on the given data

    Parameters
    ----------
    train_df : pandas dataframe
        Dataframe containing the training data
    centroids : list
        List of tuples containing the initial centroids
    distance_metric : str
        Distance metric to be used for clustering

    Returns
    -------
    None
    """
    # create points dict
    pred_cols = sorted(list(set(train_df.columns) - set('C')))
    points = {k:tuple([int(i) for i in v]) for k,v in zip(train_df['C'].values.tolist(), train_df[pred_cols].apply(tuple, axis=1).values.tolist())}
    old_centroids = [tuple([float(i) for i in j]) for j in centroids]
    centroid_size = len(old_centroids[0])
    while(True):
        # old_centroids = centroids
        new_clusters = {f'C{i+1}':[] for i in range(len(old_centroids))}
        for k, p in points.items():
            d = []
            for idx, q in enumerate(old_centroids):
                if distance_metric == 'e2':
                    d.append((euclidean_distance(p,q), idx+1))
                if distance_metric == 'manh':
                    d.append((manhattan_distance(p,q), idx+1))
            new_centre = min(d)[1]
            new_clusters[f'C{new_centre}'].append((k,p))

        new_centroids = []
        new_clusters = {i: new_clusters[i] for i in sorted(list(new_clusters.keys()))}

        for idx, (k,v) in enumerate(new_clusters.items()):
            d = []
            if len(v) == 0:
                # take old centroid
                new_centroids.append(old_centroids[idx])
            else:
                for i in v:
                    d.append(i[1])
                point_ = []
                for j in range(centroid_size):
                    point_.append(np.average([i[j] for i in d]))
                new_centroids.append(tuple(point_))
        
        # new_centroids = centroids
        
        if (old_centroids == new_centroids):
            break
        else:
            old_centroids = new_centroids

    for k,v in new_clusters.items():
        v_ = ','.join([j[0] for j in v])
        out = f"{k} = " + "{" + f"{v_}" + "}"
        print(out)
    for c in new_centroids:
        c_ = " ".join([str(i) for i in c])
        out = "([" + c_ + "])"
        print(out)



def main():

    # algorithm flags
    kNN_ = False
    naive_bayes_ = False
    kMeans_ = False

    # initialize variables
    VERBOSE = False
    kNN_k = None
    nb_delta = None
    C_flag = None
    dist = None
    centroids = None

    # Parse command line arguments
    if len(sys.argv) < 2:
        print(f"Insufficient arguments")
        exit()
        
    args = [i for i in sys.argv[1:]]

    if '-v' in args:
        VERBOSE = True
        args = [e for e in args if e not in ('-v')]

    if '-train' in args:
        try:
            train_filename = args[args.index('-train') + 1]
            args = [e for e in args if e not in ('-train', train_filename)]
        except:
            log.error(f"Invalid arguments: -train <filename> missing")
            log.error(f"Exiting...")
            exit()

    if '-test' in args:
        try:
            test_filename = args[args.index('-test') + 1]
            args = [e for e in args if e not in ('-test', test_filename)]
        except:
            log.error(f"Invalid arguments: -test <filename> missing")
    else:
        ## KMeans algorithm => no test file
        kMeans_ = True
        if '-d' in args:
            dist = args[args.index('-d') + 1]
            if dist not in ('e2', 'manh'):
                log.error(f"Invalid arguments: -d <distance_metric> must be 'e2' or 'manh'")
                exit()
            args = [e for e in args if e not in ('-d', dist)]
        _centroids = [i for i in args]
        centroids = [tuple(int(j) for j in (i.split(','))) for i in args]
        args = [e for e in args if e not in _centroids]


    if '-K' in args:
        try:
            kNN_ = True
            k = args[args.index('-K') + 1]
            try:
                if int(k) > 0:
                    kNN_ = True
                    kNN_k = int(k)
                if int(k) == 0:
                    naive_bayes_ = True
            except:
                log.error(f"Invalid arguments: -K <int> must be positive or zero")
            args = [i for j, i in enumerate(args) if j not in (args.index('-K') + 1, args.index('-K'))]
        except:
            log.error(f"Invalid arguments: -K <int> missing")

    if '-C' in args:
        try:
            naive_bayes_ = True
            delta = args[args.index('-C') + 1]
            try:
                if int(delta) > 0:
                    C_flag = True
                    nb_delta = int(delta)
                if int(delta) == 0:
                    C_flag = False
            except:
                log.error(f"Invalid arguments: -C <int> must be positive or zero")
            args = [i for j, i in enumerate(args) if j not in (args.index('-C') + 1, args.index('-C'))]
        except:
            log.error(f"Invalid arguments: -C <int> missing")
    
    if kNN_k and nb_delta:
        if kNN_k > 0 and nb_delta > 0:
            # K and C cannot be both positive. Defaulting to Naive Bayes with delta = 0
            kNN_ = False
            naive_bayes_ = True
            C_flag = False
            nb_delta = 0
            kNN_k = None

    
    if naive_bayes_:
        train_df, test_df = read_data(train_filename), read_data(test_filename)
        assert train_df.columns.values.tolist() == test_df.columns.values.tolist(), "Train and test data must have same attributes"
        metrics = naive_bayes(train_df=train_df, test_df=test_df, C_flag=C_flag, delta=nb_delta, verbose=VERBOSE)
        print_metrics(metrics=metrics)
        exit()
    
    if kNN_:
        train_df, test_df = read_data(train_filename), read_data(test_filename)
        assert train_df.columns.values.tolist() == test_df.columns.values.tolist(), "Train and test data must have same attributes"
        metrics = kNN(train_df=train_df, test_df=test_df, kNN_k=kNN_k, verbose=VERBOSE)
        print_metrics(metrics=metrics)
        exit()

    if kMeans_:
        train_df = read_data(train_filename)
        kMeans(train_df=train_df, centroids=centroids, distance_metric=dist)
        exit()
    

if __name__ == '__main__':
    main()