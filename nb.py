import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data: Text messages and their corresponding labels (spam or not spam)
data = {
    'text': [
        "Congratulations, you've won a free ticket! Call now to claim.",
        "Meeting tomorrow at 10am. Let me know if you can make it.",
        "URGENT! Your account has been compromised. Respond immediately.",
        "Hey, are we still on for lunch today?",
        "Win a brand new car! Just click the link and enter your details."
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert labels to binary values (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Preprocess text data using Bag of Words
vectorizer = CountVectorizer()  # You can also try TfidfVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the model
nb_classifier.fit(X_train_bow, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_bow)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
