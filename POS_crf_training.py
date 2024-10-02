import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from timeit import default_timer as timer
import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
import copy

from nltk.corpus import brown

start = timer()

# downloading brown corpus data
nltk.download('brown')
nltk.download("universal_tagset")
nltk.download("punkt")

mean = lambda l : sum(l) / len(l) # computes the mean of a list

def buildTags(data):
    # Compute the set of unique tags present in the data
    tags = set()
    for s in data:
        for w, t in s:
            tags.add(t)
    tags = sorted(list(tags))
    return tags

def createFolds(data, k):
    # Randomly shuffles the data and returns k equal sized segments for k-fold cross validation
    n = len(data)
    n_split = n // k # approx split size
    # random.shuffle(data)
    folds = []
    for start in range(0, n, n_split):
        end = min(start + n_split, n)
        fold = data[start : end]
        folds.append(fold)
    return folds

def printClasswise(metrics):
    # Pretty-printer for the classwise metrics
    for i in range(N_TAGS):
        print(f"{TAGS[i]}: {metrics[i]:.3f}", end = ", ")
    print()

corpus = list(brown.tagged_sents(tagset="universal"))
# corpus = corpus[0:1000]
# print(corpus)

TAGS = buildTags(corpus)
N_TAGS = len(TAGS)

def plotCM(cm, name):
    # Plots the confusion matrix provided as a numpy array
    # df_cm = pd.DataFrame(cm, index = [t for t in TAGS], columns = [t for t in TAGS])
    df_cm = pd.DataFrame(cm, index = [t for t in TAGS], columns = [t for t in TAGS])
    plt.figure(figsize = (15, 10))
    sns.heatmap(df_cm, annot = True, cmap = plt.cm.Blues)
    # plt.show()
    plt.savefig(f"{name}")

# Define a function to extract features for each word in a sentence
def word_features(sentence, i):
	word = sentence[i][0]
	features = {
		'word': word,
		'is_first': i == 0, #if the word is a first word
		'is_last': i == len(sentence) - 1, #if the word is a last word
		'is_capitalized': word[0].upper() == word[0],
		'is_all_caps': word.upper() == word,	 #word is in uppercase
		'is_all_lower': word.lower() == word,	 #word is in lowercase
		#prefix of the word
		'prefix-1': word[0], 
		'prefix-2': word[:2],
		'prefix-3': word[:3],
		#suffix of the word
		'suffix-1': word[-1],
		'suffix-2': word[-2:],
		'suffix-3': word[-3:],
		#extracting previous word
		'prev_word': '' if i == 0 else sentence[i-1][0],
		#extracting next word
		'next_word': '' if i == len(sentence)-1 else sentence[i+1][0],
		'has_hyphen': '-' in word, #if word has hypen
		'is_numeric': word.isdigit(), #if word is in numeric
		'capitals_inside': word[1:].lower() != word[1:]
	}
	return features

random.shuffle(corpus)
# folds = createFolds(corpus, k = 5)

# Extract features for each sentence in the corpus
X = []
y = []
for sentence in corpus:
    X_sentence = []
    y_sentence = []
    for i in range(len(sentence)):
        X_sentence.append(word_features(sentence, i))
        y_sentence.append(sentence[i][1])
    X.append(X_sentence)
    y.append(y_sentence)

## We need to loop from here
print(f"\n\n\n len X is {len(X)} \n\n\n")

X_folds = createFolds(X, 5)
y_folds = createFolds(y, 5)

# print(f"FOLD 1 size : {len(X_folds[0])}, {len(y_folds[0])}")
# print(f"FOLD 2 size : {len(X_folds[1])}, {len(y_folds[1])}")
# print(f"FOLD 3 size : {len(X_folds[2])}, {len(y_folds[2])}")
# print(f"FOLD 4 size : {len(X_folds[3])}, {len(y_folds[3])}")
# print(f"FOLD 5 size : {len(X_folds[4])}, {len(y_folds[4])}\n")

# print(f"data type y is {type(y_folds[0][0][0])}")
# print(f"data type y is {type(y_folds[1][0][0])}")
# print(f"data type y is {type(y_folds[2][0][0])}")
# print(f"data type y is {type(y_folds[3][0][0])}")
# print(f"data type y is {type(y_folds[4][0][0])}\n")
# print(f"\nFOLDS:\n\n{X_folds}\n\n{y_folds}")

avg_cm = 0 # ignoring the dummy tag added at the start of each sentence

# Initialize average metrics
avg_acc = 0
avg_prec = 0
avg_rec = 0
avg_f1 = 0
avg_f05 = 0
avg_f2 = 0

# Average tagwise metrics
avg_prec_tagwise = 0
avg_rec_tagwise = 0
avg_f1_tagwise = 0

for fv in range(5):
    start_fv = timer()
    print(f"----------------------------------")
    print(f"----------FOLD {fv+1}-------------")
    print(f"----------------------------------")

    X_train = []
    for ft in range(5):
        if ft != fv:
            X_train += copy.deepcopy(X_folds[ft])

    y_train = []
    for ft in range(5):
        if ft != fv:
            y_train += copy.deepcopy(y_folds[ft])

    # Split the data into training and testing sets
    # split = int(0.8 * len(X))
    # X_train = X[:split]
    # y_train = y[:split]
    X_test = []
    y_test = []
    X_test = copy.deepcopy(X_folds[fv])
    y_test = copy.deepcopy(y_folds[fv])

    # print(f"\nThe X_test is :\n{X_test[:5]}\n")

    # print(f"\nThe Y_test is : \n{y_test[:5]}\n")

    # t = input("Do you want to train again? (YES = 1, NO = 0)\n")

    # if t == 1:
    # Train a CRF model on the training data
    # print(f"This is type on fold {fv+1} : {type(y_train[0][0])}\n")
    # print(f"This is type on fold {fv+1} : {type(y_folds[0][0][0])}\n")
    # print(f"This is type on fold {fv+1} : {type(y_folds[1][0][0])}\n")
    # print(f"This is type on fold {fv+1} : {type(y_folds[2][0][0])}\n")
    # print(f"This is type on fold {fv+1} : {type(y_folds[3][0][0])}\n")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    with open(f'trained_crf_{fv}.pkl', 'wb') as clf:
        pickle.dump(crf, clf)

    # with open('trained_crf.pkl', 'rb') as of:
    #     crf = pickle.load(of)

    id2tag = lambda i : TAGS[i]
    tagmap = {TAGS[i]:i for i in range(N_TAGS)}
    tag2id = lambda i : tagmap[i]

    # Make predictions on the test data and evaluate the performance
    y_pred = crf.predict(X_test)
    y_pred_list = y_pred.tolist()
    # print(f"y_pred are : \n{y_pred_list}\n")

    for i in range(len(y_pred_list)):
        for j in range(len(y_pred[i])):
            y_pred[i][j] = tag2id(y_pred_list[i][j])

    # print(f"y_pred are : \n{y_pred_list}\n")

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            y_test[i][j] = tag2id(y_test[i][j])

    # Flatten y_test and y_pred_list to be 1D arrays
    y_test_flat = [tag for sentence in y_test for tag in sentence]
    y_pred_flat = [tag for sentence in y_pred_list for tag in sentence]

    # print(f"y_test flattened : \n\n{y_test_flat}\n")
    # print(f"y_pred_flattened : \n\n{y_pred_flat}\n")

    metric_labels = [i for i in range(N_TAGS)]
    cm = confusion_matrix(y_test_flat, y_pred_flat, normalize = "true", labels = metric_labels)

    acc = accuracy_score(y_test_flat, y_pred_flat)

    prec = precision_score(y_test_flat, y_pred_flat, average = None, labels = metric_labels)
    rec = recall_score(y_test_flat, y_pred_flat, average = None, labels = metric_labels)
    f1 = f1_score(y_test_flat, y_pred_flat, average = None, labels = metric_labels)

    f05 = fbeta_score(y_test_flat, y_pred_flat, beta = 0.5, average = "macro", labels = metric_labels)
    f2 = fbeta_score(y_test_flat, y_pred_flat, beta = 2, average = "macro", labels = metric_labels)

    # Tag-wise metrics
    print("Tag-wise precision:")
    printClasswise(prec)

    print("Tag-wise recall:")
    printClasswise(rec)

    print("Tag-wise F1:")
    printClasswise(f1)

    # Overall metrics for this fold
    print("Accuracy:", acc)
    print("Avg. precision:", mean(prec))
    print("Avg. recall:", mean(rec))
    print("Avg. f_0.5:", f05)
    print("Avg. f1:", mean(f1))
    print("Avg. f2:", f2)

    # Update global avg values
    avg_acc += acc
    avg_prec += mean(prec)
    avg_rec += mean(rec)
    avg_f1 += mean(f1)
    avg_f05 += f05
    avg_f2 += f2

    # Update tagwise metrics
    avg_prec_tagwise += prec
    avg_rec_tagwise += rec
    avg_f1_tagwise += f1

    # Update cm
    avg_cm += cm
    end_fv = timer()
    print(f"\n\nTIME FOR FOLD {fv+1} : {end_fv-start_fv}\n\n")

    print(f"----------------------------------")
    print(f"----------------------------------")

    plotCM(cm, f"figure_fold_{fv}")

    # print(f"Accuracy fold{fv+1} : {metrics.flat_accuracy_score(y_test, y_pred)}")

# end the looping here
# Average over the number of folds

avg_cm /= 5

avg_acc /= 5
avg_prec /= 5
avg_rec /= 5
avg_f1 /= 5
avg_f05 /= 5
avg_f2 /= 5

avg_prec_tagwise /= 5
avg_rec_tagwise /= 5
avg_f1_tagwise /= 5

# Final metrics reported

# Tag-wise metrics
print("Tag-wise precision:")
printClasswise(avg_prec_tagwise)

print("Tag-wise recall:")
printClasswise(avg_rec_tagwise)

print("Tag-wise F1:")
printClasswise(avg_f1_tagwise)

# Overall metrics for this fold
print("Accuracy:", avg_acc)
print("Avg. precision:", avg_prec)
print("Avg. recall:", avg_rec)
print("Avg. f_0.5:", avg_f05)
print("Avg. f1:", avg_f1)
print("Avg. f2:", avg_f2)

plotCM(avg_cm, f"figure_avg")

end = timer()
print(f"\n Total time taken = {end - start}")