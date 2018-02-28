import sys
import tweepy

# Credentials for tweepy (redacted)
app_key = ""
app_secret = ""
oauth_token = ""
oauth_token_secret = ""

# Authentication
auth = tweepy.OAuthHandler(app_key, app_secret)
auth.set_access_token(oauth_token, oauth_token_secret)

# The API handler
api = tweepy.API(auth)

# Trusted and fake source Twitter IDs
trusted_ids = ["51241574", "19701628", "3108351", "5392522", "14606079", "1652541", "15754281", "5988062"]
fake_ids = ["1266239359", "2275780065", "2302467404", "19222214", "2455959913", "609142275", "17793358", "735574058167734273"]

# Lists to store the tweets in
trusted_tweets = []
fake_tweets = []

'''
-Loop through each ID
-Loop through their most recent 200 tweets
-Parse the tweets' text from JSON and check its encoding (which ignores unknown characters)
-Add the text and a newline to the respective list
'''
for user_id in trusted_ids:
    for tweet in api.user_timeline(id=user_id, count=200):
        trusted_tweets.append(str(tweet._json["text"].encode(sys.stdout.encoding, errors="ignore")) + "\n")

for user_id in fake_ids:
    for tweet in api.user_timeline(id=user_id, count=200):
        fake_tweets.append(str(tweet._json["text"].encode(sys.stdout.encoding, errors="ignore")) + "\n")

# Write the lists to files
trusted_file = open("trusted.txt", "w")
trusted_file.writelines(trusted_tweets)
trusted_file.close()

fake_file = open("fake.txt", "w")
fake_file.writelines(fake_tweets)
fake_file.close()
