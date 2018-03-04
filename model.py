import nltk
from nltk.tokenize import word_tokenize
import random
from nltk.corpus import stopwords


fake_tweets = open('tweets/fake_news_tweets.txt', 'r')
real_tweets = open('tweets/trusted_news_tweets.txt', 'r')

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

word_features = list(all_words.keys())[:3000]

# determine if a frequent word is in the given file
def find_features(tweet):
    words = set(word_tokenize(tweet))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

feature_sets = [(find_features(tweet), category) \
                for (tweet, category) in categorized_tweets]

training_set = feature_sets[:1502]
testing_set = feature_sets[1502:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",\
      (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(30)








