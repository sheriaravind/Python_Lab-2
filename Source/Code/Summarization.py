import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import bigrams

def Summarize():
    with open(r'C:\Users\aravi\OneDrive\Desktop\Input.txt', 'r') as f:
        data = f.read()
        print("Data from the input file: \n\n", data)
    data_word = word_tokenize(data) # Word Tokanizer
    data_sent = sent_tokenize(data) # Sentence Tokanizer
    lemmatizer = WordNetLemmatizer()
    data_lemmatized = []
    for word in data_word:
        fr_lema = lemmatizer.lemmatize(word.lower()) # Lemmatize the words from tokanized words
        data_lemmatized.append(fr_lema)

    print("Lemmatized Data: \n\n", data_lemmatized, "\n")
    bigram_data = []
    for grams in bigrams(data_lemmatized): # Forms the bigrams from the lemmatized data
        bigram_data.append(grams)
    print("Bigram Data: \n\n", bigram_data, "\n")
    fdist1 = nltk.FreqDist(bigram_data) # Calculates the frequency of the bigram_data
    bigram_freq = fdist1.most_common() #Finds the most common
    print("Bigrams with Frequency: \n",bigram_freq,"\n")
    top_five = fdist1.most_common(5) # Finds the to 5 common elements
    print("Top five Bigrams with frequency: \n",top_five,"\n")
    rep_sent1 = []
    for sent in data_sent:
        for word, words in bigram_data:
            for ((s, t), l) in top_five:
                if (word, words == s, t):
                    rep_sent1.append(sent)
    print("Summarized Data \n")
    print(max(rep_sent1, key=len))

if __name__ == '__main__':
    Summarize()