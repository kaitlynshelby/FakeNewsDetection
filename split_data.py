import random

fake_tweets = open('cleaned_tweets/fake_tweets.txt', 'r')
real_tweets = open('cleaned_tweets/trusted_tweets.txt', 'r')

new_real = open('split_real_tweets.txt', 'w')
new_fake = open('split_fake_tweets.txt', 'w')
testing = open('testing_data.txt', 'w')

fake_list = [tweet for tweet in fake_tweets.readlines()]
real_list = [tweet for tweet in real_tweets.readlines()]

random.shuffle(fake_list)
random.shuffle(real_list)


for tweet in real_list[:-50]:
    new_real.write(tweet)

for tweet in fake_list[:-50]:
    new_fake.write(tweet)

for tweet in real_list[-50:]:
    testing.write(tweet)

for tweet in fake_list[-50:]:
    testing.write(tweet)



fake_tweets.close()
real_tweets.close()
new_real.close()
new_fake.close()
testing.close()
