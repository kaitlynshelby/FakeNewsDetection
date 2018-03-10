import nltk
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import random
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


fake_tweets = open('cleaned_tweets/fake_tweets.txt', 'r')
real_tweets = open('cleaned_tweets/trusted_tweets.txt', 'r')

# categorize tweets
categorized_tweets = [(tweet, 'fake') for tweet in fake_tweets.readlines()] + \
            [(tweet, 'real') for tweet in real_tweets.readlines()]

random.shuffle(categorized_tweets)

fake_tweets.seek(0)
real_tweets.seek(0)

all_words = []


# ignore uninteresting words
stopwords = set(stopwords.words('english'))

for line in fake_tweets.readlines():
    for t in word_tokenize(line):
        if t not in stopwords:
            all_words.append(t.lower())

for line in real_tweets.readlines():
    for t in word_tokenize(line):
        if t not in stopwords:
            all_words.append(t.lower())


fake_tweets.close()
real_tweets.close()

# find most frequent words 
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3500]


# determine if a frequent word is in the given file
def find_features(tweet):
    words = set(word_tokenize(tweet))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


training_set = categorized_tweets[:5505]
testing_set = categorized_tweets[5505:]

training_features = [(find_features(tweet), category) for (tweet, category) in training_set] 
testing_features = [(find_features(tweet), category) for (tweet, category) in testing_set]

# classifiers and acronyms

classifiers = [(nltk.NaiveBayesClassifier, "nb"),
               (SklearnClassifier(MultinomialNB()), "mnb"),
               (SklearnClassifier(BernoulliNB()), "bnb"),
               (SklearnClassifier(LogisticRegression()), "lr"),
               (SklearnClassifier(SGDClassifier()), "sgd"),
               (SklearnClassifier(LinearSVC()), "lsvc"),
               (SklearnClassifier(NuSVC()), "nsvc")]

# train and save classifiers 

for c in classifiers:
    train = c[0].train(training_features)
    with open("saved_classifiers/" + c[1] + ".pickle", "wb") as output_file:
        pickle.dump(train, output_file)

# load classifiers
def load_classifier(acronym):
    with open("saved_classifiers/" + acronym + ".pickle", "rb") as input_file:
        load_f = pickle.load(input_file)
    return load_f


nb_classifier = load_classifier("nb")
mnb_classifier = load_classifier("mnb")
bnb_classifier = load_classifier("bnb")
lr_classifier = load_classifier("lr")
sgd_classifier = load_classifier("sgd")
lsvc_classifier = load_classifier("lsvc")
nsvc_classifier = load_classifier("nsvc")

voted_classifier = VoteClassifier(nb_classifier,
                                  mnb_classifier,
                                  bnb_classifier,
                                  lr_classifier,
                                  sgd_classifier,
                                  lsvc_classifier,
                                  nsvc_classifier)

# print accuracies
print("Accuracy percent:", 
      (nltk.classify.accuracy(voted_classifier, testing_features))*100)
print()

# give each tweet an individual score
tagged_tweets = []

for (tweet, category) in testing_set:
    tweet_features = find_features(tweet)
    auth = voted_classifier.classify(tweet_features)
    print(tweet, '\n\t\t', auth.upper(), '\t', 
           'confidence: ', voted_classifier.confidence(tweet_features)*100)
    print()
    tagged_tweets.append(tuple([tweet, auth]))
