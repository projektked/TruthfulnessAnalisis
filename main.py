import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime

nltk.download('vader_lexicon')
sentimentAnalyzer = SentimentIntensityAnalyzer()

df = pd.read_csv('')
df = df.drop(columns=['author', 'over_18', 'time_created', 'subreddit'])
df['date_created'] = df['date_created'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())

df['sentiment'] = df['title'].apply(lambda x: sentimentAnalyzer.polarity_scores(x)['compound'])

df = df.groupby(df.date_created.map(lambda x: x.day)).mean()

plt.plot(df['date_created'], df['sentiment'])

plt.xlabel('sentiment')
plt.ylabel('date')

plt.show()
