import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Download dos recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english') and token.isalpha()]
    return " ".join(cleaned_tokens)


df = pd.read_excel('dbmBasesequalizadas.xlsx')
df['text'] = (df['title'] + " " + df['description']).apply(preprocess_text)


train_df, val_df = train_test_split(df, test_size=0.3)  # Removed random_state

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])

# Modelo de classificação
model = MultinomialNB()
model.fit(X_train, train_df['classificaçãoDadoPessoal'])

# Avaliação
predictions = model.predict(X_val)
print(classification_report(val_df['classificaçãoDadoPessoal'], predictions))

# Matriz de confusão
cm = confusion_matrix(val_df['classificaçãoDadoPessoal'], predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - NLTK Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()