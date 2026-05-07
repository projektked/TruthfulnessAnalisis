import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from wordcloud import STOPWORDS
from wordcloud import WordCloud

nltk.download('vader_lexicon')
sentimentAnalyzer = SentimentIntensityAnalyzer()

df = pd.read_csv('')
df = df.drop(columns=['author', 'over_18', 'time_created', 'subreddit'])

df['sentiment'] = df['title'].apply(lambda x: sentimentAnalyzer.polarity_scores(x)['compound'])

text = ' '.join(df['title'].astype(str).tolist())
text = re.sub(r'[^A-Za-z\s]', '', text)
text = text.lower()

stopwords = set('s').union(STOPWORDS)
text = ' '.join(word for word in text.split() if word not in stopwords)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Wordcloud of data")
plt.show()

text_arr = text.split()
text_arr_filtered = list(set(text_arr))
word_polartity_dict = dict()

for x in text_arr:
    word_polartity_dict[x]=sentimentAnalyzer.polarity_scores(x)['compound']



print(word_polartity_dict)
# if df['sentiment'].mean()>0:
#
# else if df['sentiment'].mean()<0:
