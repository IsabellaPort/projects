# projects

#Sentiment Analysis on thesis examination 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Sample responses
responses = [
    "Yes, I have used dropshipping before. It's a business model where the seller doesn't keep the products in stock, but instead transfers the customer orders and shipment details to either the manufacturer or a wholesaler, who then ships the products directly to the customer.",
    "Dropshipping is a supply chain management method in which the retailer does not keep goods in stock but instead transfers the customer orders and shipment details to either the manufacturer, another retailer, or a wholesaler, who then ships the goods directly to the customer.",
    "No, I'm not familiar with dropshipping.",
    "Yes, I have heard about dropshipping. It's a business model where the seller doesn't keep the products in stock. Instead, when a seller makes a sale, they purchase the item from a third party and have it shipped directly to the customer.",
    "Dropshipping is a retail fulfillment method where a store doesn't keep the products it sells in stock. Instead, when a store sells a product, it purchases the item from a third party and has it shipped directly to the customer. As a result, the merchant never sees or handles the product.",
    "I haven't used dropshipping myself, but I know it's a business model where the seller doesn't keep the products in stock. Instead, they partner with a supplier who handles the inventory and shipping."
]

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return words

# Tokenize and preprocess all responses
processed_responses = [preprocess_text(response) for response in responses]

# Count word frequency
word_count = Counter()
for words in processed_responses:
    word_count.update(words)

# Print the most common words
print("Most common words:")
for word, count in word_count.most_common(10):
    print(f"{word}: {count}")

# Plot word frequency
plt.figure(figsize=(10, 6))
common_words = [word for word, count in word_count.most_common(10)]
common_counts = [count for word, count in word_count.most_common(10)]
plt.bar(common_words, common_counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
