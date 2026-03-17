# 🚀 Text Generation using LSTM (Full Code)

# Install TensorFlow (only needed in Colab)
!pip install tensorflow -q

# 📚 Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 📝 Dataset (you can expand this)
text = """
deep learning is fun
deep learning is powerful
machine learning is amazing
artificial intelligence is the future
"""

# 🔤 Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# 📏 Padding
max_len = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# 🎯 Split data
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 🤖 Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 50, input_length=max_len-1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 📊 Model Summary
model.summary()

# 🏋️ Train Model
model.fit(X, y, epochs=200, verbose=1)

# 🔮 Text Generation Function
def generate_text(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
                
    return seed_text

# 🧪 Test
print(generate_text("deep learning is", 5))