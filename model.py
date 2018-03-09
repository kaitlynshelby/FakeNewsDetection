import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import pickle


fake_tweets = open('cleaned_tweets/fake_tweets.txt', 'r')
real_tweets = open('cleaned_tweets/trusted_tweets.txt', 'r')

# categorize tweets
categorized_tweets = [((list(word_tokenize(tweet))), 'fake') for tweet in fake_tweets.readlines()] + \
            [((list(word_tokenize(tweet))), 'real') for tweet in real_tweets.readlines()]

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

# find most frequent words longer than 1 character
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3500]


# determine if a frequent word is in the given file
def find_features(tweet):
    words = set(tweet)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


training_set = categorized_tweets[:5505]
testing_set = categorized_tweets[5505:]

training_features = [(find_features(tweet), category) for (tweet, category) in training_set] 
testing_features = [(find_features(tweet), category) for (tweet, category) in testing_set] 


# run the data sets through the NaiveBayes classifier and print results
classifier = nltk.NaiveBayesClassifier.train(training_features)

print("Classifier accuracy percent:",
      (nltk.classify.accuracy(classifier, testing_features))*100)

# classifier.show_most_informative_features(30)


# give each tweet an individual score - work in progress
tagged_tweets = []

labels = ['real', 'fake']

for (tweet, category) in testing_set:
    text = " ".join(tweet)
    auth = classifier.prob_classify(find_features(tweet))
    print(text + '\n\t\tPROB REAL: ' + str("%.2f" % (auth.prob('real') * 100)) + ' PROB FAKE: ' + str("%.2f" % (auth.prob('fake') * 100)))
    print()
    tagged_tweets.append(tuple([text, auth]))

