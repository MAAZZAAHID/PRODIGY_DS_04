import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

df = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")




df = df[['tweet', 'label']]
df.columns = ['text', 'sentiment']
df['sentiment'] = df['sentiment'].map({0: 'Negative', 1: 'Positive'})

plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()





def generate_wordcloud(sentiment):
    text = " ".join(df[df['sentiment'] == sentiment]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Most Common Words in {sentiment} Tweets")
    plt.show()



generate_wordcloud('Positive')
generate_wordcloud('Negative')

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 'Positive' if polarity > 0 else 'Negative'

df['predicted_sentiment'] = df['text'].apply(analyze_sentiment)


accuracy = (df['sentiment'] == df['predicted_sentiment']).mean()
print(f"TextBlob Sentiment Analysis Accuracy: {accuracy:.2%}")