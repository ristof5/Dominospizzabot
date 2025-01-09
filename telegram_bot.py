import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Load dataset
try:
    dataset = pd.read_excel("dominos.xlsx", engine='openpyxl')  # Menggunakan read_excel
    print("Dataset Loaded Successfully!")
    print("Kolom Dataset:", dataset.columns)  # Debug: Lihat nama kolom
    print(dataset.head())  # Debug: Tampilkan beberapa baris data
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Rename columns if they are incorrect
if 'Pertanyaan' not in dataset.columns or 'Jawaban' not in dataset.columns:
    print("Kolom tidak ditemukan, mengubah nama kolom...")
    dataset.columns = ['Pertanyaan', 'Jawaban']
    print("Kolom setelah rename:", dataset.columns)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset['Pertanyaan'])

# Function to get the most similar response or provide a default message
def get_response(user_input):
    user_tfidf = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_tfidf, tfidf_matrix)
    max_similarity = similarity.max()  # Check the highest similarity score
    if max_similarity > 0.3:  # Set a threshold for similarity
        idx = similarity.argmax()
        return dataset.iloc[idx]['Jawaban']
    else:
        return "Maaf, saya tidak dapat menemukan jawaban untuk pertanyaan Anda. Silakan coba pertanyaan lain atau hubungi layanan pelanggan kami."

# Load API Token
API_TOKEN = '8048423270:AAFOGJt370ji1KvIIiIPDcvERBVWN6Bg9Cw'

# Define start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Halo! Saya adalah chatbot Domino's Pizza. Tanyakan apa saja tentang Domino's Pizza!")

# Define message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text
    # Get response from TF-IDF model
    response = get_response(user_input)
    await update.message.reply_text(response)

# Main function to start the bot
def main():
    application = ApplicationBuilder().token(API_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
