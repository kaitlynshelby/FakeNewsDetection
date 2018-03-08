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
trusted_ids = open("account_ids/trusted_ids.txt").read().split(",")
fake_ids = open("account_ids/fake_ids.txt").read().split(",")

# Lists to store the tweets in
trusted_tweets = []
fake_tweets = []

'''
-Loop through each ID and print their name and screen name to the console
-Loop through their most recent 200 tweets
-Parse the tweets' full text from JSON and check its encoding (which ignores unknown characters)
-Add the text and a newline to the respective list
'''

print("Trusted:")
for user_id in trusted_ids:
    print(api.get_user(user_id)._json["name"] + " - " + api.get_user(user_id)._json["screen_name"])

    for tweet in api.user_timeline(id=user_id, count=200, tweet_mode="extended"):
        try:
            trusted_tweets.append(str(tweet._json["retweeted_status"]["full_text"]
                                      .encode(sys.stdout.encoding, errors="ignore"))[2:-1] + "\n")
        except:
            trusted_tweets.append(str(tweet._json["full_text"]
                                      .encode(sys.stdout.encoding, errors="ignore"))[2:-1] + "\n")

print("\nFake:")
for user_id in fake_ids:
    print(api.get_user(user_id)._json["name"] + " - " + api.get_user(user_id)._json["screen_name"])

    for tweet in api.user_timeline(id=user_id, count=200, tweet_mode="extended"):
        try:
            fake_tweets.append(str(tweet._json["retweeted_status"]["full_text"]
                                      .encode(sys.stdout.encoding, errors="ignore"))[2:-1] + "\n")
        except:
            fake_tweets.append(str(tweet._json["full_text"]
                                      .encode(sys.stdout.encoding, errors="ignore"))[2:-1] + "\n")

# Write the lists to files
trusted_file = open("raw_tweets/trusted_tweets.txt", "w")
trusted_file.writelines(trusted_tweets)
trusted_file.close()

fake_file = open("raw_tweets/fake_tweets.txt", "w")
fake_file.writelines(fake_tweets)
fake_file.close()
