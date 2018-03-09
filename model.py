import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
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
#stopwords = set(stopwords.words('english'))

for line in fake_tweets.readlines():
    for t in word_tokenize(line):
        #if t not in stopwords:
            all_words.append(t.lower())

for line in real_tweets.readlines():
    for t in word_tokenize(line):
        #if t not in stopwords:
            all_words.append(t.lower())


fake_tweets.close()
real_tweets.close()

# find most frequent words longer than 1 character
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

# train and save all classifiers
nb_classifier = nltk.NaiveBayesClassifier.train(training_features)
save_classifier = open("nb.pickle","wb")
pickle.dump(nb_classifier, save_classifier)
save_classifier.close()

mnb_classifier = SklearnClassifier(MultinomialNB())
mnb_classifier.train(training_features)
save_classifier = open("mnb.pickle","wb")
pickle.dump(mnb_classifier, save_classifier)
save_classifier.close()

bnb_classifier = SklearnClassifier(BernoulliNB())
bnb_classifier.train(training_features)
save_classifier = open("bnb.pickle","wb")
pickle.dump(bnb_classifier, save_classifier)
save_classifier.close()

lr_classifier = SklearnClassifier(LogisticRegression())
lr_classifier.train(training_features)
save_classifier = open("lr.pickle","wb")
pickle.dump(lr_classifier, save_classifier)
save_classifier.close()

sgd_classifier = SklearnClassifier(SGDClassifier())
sgd_classifier.train(training_features)
save_classifier = open("sgd.pickle","wb")
pickle.dump(sgd_classifier, save_classifier)
save_classifier.close()

#svc_classifier = SklearnClassifier(SVC())
#svc_classifier.train(training_features)

lsvc_classifier = SklearnClassifier(LinearSVC())
lsvc_classifier.train(training_features)
save_classifier = open("lsvc.pickle","wb")
pickle.dump(lsvc_classifier, save_classifier)
save_classifier.close()

nsvc_classifier = SklearnClassifier(NuSVC())
nsvc_classifier.train(training_features)
save_classifier = open("nsvc.pickle","wb")
pickle.dump(nsvc_classifier, save_classifier)
save_classifier.close()



# load classifiers
classifier_f = open("nb.pickle", "rb")
nb_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("mnb.pickle", "rb")
mnb_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("bnb.pickle", "rb")
bnb_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("lr.pickle", "rb")
lr_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("sgd.pickle", "rb")
sgd_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("lsvc.pickle", "rb")
lsvc_classifier = pickle.load(classifier_f)
classifier_f.close()

classifier_f = open("nsvc.pickle", "rb")
nsvc_classifier = pickle.load(classifier_f)
classifier_f.close()

voted_classifier = VoteClassifier(nb_classifier,
                                  mnb_classifier,
                                  bnb_classifier,
                                  lr_classifier,
                                  sgd_classifier,
                                  lsvc_classifier,
                                  nsvc_classifier)




# print accuracies
print("Naive Bayes classifier accuracy percent:",
      (nltk.classify.accuracy(nb_classifier, testing_features))*100)

nb_classifier.show_most_informative_features(15)

print("MNB classifier accuracy percent:",
      (nltk.classify.accuracy(mnb_classifier, testing_features))*100)

print("BernoulliNB classifier accuracy percent:",
      (nltk.classify.accuracy(bnb_classifier, testing_features))*100)

print("LogisticRegression classifier accuracy percent:",
      (nltk.classify.accuracy(lr_classifier, testing_features))*100)

print("SGD classifier accuracy percent:",
      (nltk.classify.accuracy(sgd_classifier, testing_features))*100)

#print("SVC classifier accuracy percent:",
#      (nltk.classify.accuracy(svc_classifier, testing_features))*100)

print("LinearSVC classifier accuracy percent:",
      (nltk.classify.accuracy(lsvc_classifier, testing_features))*100)

print("NuSVC classifier accuracy percent:",
      (nltk.classify.accuracy(nsvc_classifier, testing_features))*100)

print("Voted classifier accuracy percent:", 
      (nltk.classify.accuracy(voted_classifier, testing_features))*100)



# give each tweet an individual score
tagged_tweets = []


for (tweet, category) in testing_set:
    auth = voted_classifier.classify(find_features(tweet))
    print(tweet + '\n\t\t' + auth)
    print()
    tagged_tweets.append(tuple([tweet, auth]))

