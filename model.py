import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random


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

# find most frequent words longer than 1 character
freq_dist = nltk.FreqDist(all_words).most_common(3000)

word_features = []
for w in freq_dist:
    if len(w[0]) > 1:
        word_features.append(w[0])


# determine if a frequent word is in the given file
def find_features(tweet):
    words = set(word_tokenize(tweet))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# build the training and testing data
feature_sets = [(find_features(tweet), category)
                for (tweet, category) in categorized_tweets]

training_set = feature_sets[:2750]
testing_set = feature_sets[2750:-100]

# run the data sets through the NaiveBayes classifier and print results
classifier = nltk.NaiveBayesClassifier.train(training_set)

#print("Classifier accuracy percent:",
#      (nltk.classify.accuracy(classifier, testing_set))*100)

#classifier.show_most_informative_features(30)


# give each tweet an individual score - work in progress

tagged_test_set = feature_sets[-100:]

print(training_set[0:5])
print(testing_set[0:5])
print(tagged_test_set[0:5])

tagged_tweets = []

for tweet in tagged_test_set:
    text = " ".join(tweet[0].keys())
    auth = classifier.classify(find_features(text))
    print(text, auth)
    #tagged_tweets.append(tuple([text, auth]))

#print(tagged_tweets)
