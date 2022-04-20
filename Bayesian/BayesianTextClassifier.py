from math import log
import os
from collections import Counter

# Created By Rikard Radovac 2022-04-20
# Program is based on Bayes Naive text classification and is here restricted to positive or negative reviews

INTERPUNCTIONS = "Interpunction.txt"
REMOVE_WORDS = "Removewords.txt"
TRAININGPATH = "Trainingdata"
TESTPATH = "Testdata"

def read_file(filename):
    """Reads and returns all non-empty lines from a text file

    Args:
        filename (txt): txt file

    Returns:
        list: list of lines
    """
    with open(filename) as file:
        lines = [line.rstrip() for line in file if line.rstrip()]
    return lines


def clean_text(text: str, interpunctions: list, stopwords: list):
    """Cleans up the text based on given interpunctions and stopwords

    Args:
        text (str): _description_
        interpunctions (list): interpunctions to strip from the text
        stopwords (list): words to remove from the text (counted as unnecessary)

    Returns:
        list: returns a list of all the separated words
    """
    interpunctions = "".join(interpunctions)
    splitted_text = text.lower().split()
    words = []
    for word in splitted_text:
        word = word.strip(interpunctions)
        if word in stopwords or not word:
            continue
        words.append(word)
    return words


def create_classification(dirname: dir, interpunctions: list, stopwords: list):
    """Creates a text classification based on Bayes Naive classification, with positive and negative as outputs

    Args:
        dirname (dir): directory of files to train
        interpunctions (list): interpunctions to strip from the text
        stopwords (list): words to remove from the text (counted as unnecessary)

    Returns: 
        positive_probabilities (dict) : dictionary of all trained words and their probabilities for being positive
        negative_probabilities (dict) : dictionary of all trained words and their probabilities for being negative
        positive_prior (float) : float value of a prior possibility for being positive (logarithmic scale)
        negative_prior (float) : float value of a prior possibility for being negative (logarithmic scale)
    """
    data_path = os.listdir(dirname)
    word_list = []  
    
    positive_words = []
    negative_words = []
    positives_count = 0
    negatives_count = 0
    
    for file in data_path:
        filepath = os.path.join(dirname, file)
        if "positive" in file:
            positives_count += 1
            file_state = True  # Indicates that the file is labeled as positive
        else:
            negatives_count += 1
            file_state = False  # Indicates that the file is labeled as negative 
            
        with open(filepath) as text_file:
            for line in text_file:
                splitted_text = clean_text(line, interpunctions, stopwords)

                for word in splitted_text:
                    if file_state:
                        positive_words.append(word)
                    else:
                        negative_words.append(word)
                        
                    word_list.append(word)

    positive_words_counter = Counter(positive_words)
    negative_words_counter = Counter(negative_words)
    positive_probabilities = dict()
    negative_probabilities = dict()
    distinct_words = set(word_list)
    
    for word in distinct_words:
        
        # Positive words probability using a logarithmic scale
        if word in positive_words_counter:
            probability = log(positive_words_counter[word] / (len(distinct_words) + len(positive_words)))
            positive_probabilities[word] = probability
        else:
            probability = log(1 / (len(distinct_words) + len(positive_words)))
            positive_probabilities[word] = probability
            
        # Negative words probability using a logarithmic scale
        if word in negative_words_counter:
            probability = log(negative_words_counter[word] / (len(distinct_words) + len(negative_words)))
            negative_probabilities[word] = probability
        else:
            probability = log(1 / (len(distinct_words) + len(negative_words)))
            negative_probabilities[word] = probability

    positive_prior = log(positives_count / (positives_count + negatives_count))
    negative_prior = log(1 - positive_prior)
    
    return positive_probabilities, negative_probabilities, positive_prior, negative_prior



def classify(dirname: dir, interpunctions: list, stopwords: list,
            positive_probabilites: dict, negative_probabilities: dict, positive_prior: float, negative_prior: float):
    """Classifies the files in the directory based on trained data (positive or negative in this case)

    Args:
        dirname (dir): name of the directory to classify
        interpunctions (list): interpunctions to strip from the text
        stopwords (list): words to remove from the text (counted as unnecessary)
        positive_probabilities (dict) : dictionary of all trained words and their probabilities for being positive
        negative_probabilities (dict) : dictionary of all trained words and their probabilities for being negative
        positive_prior (float) : float value of a prior possibility for being positive (logarithmic scale)
        negative_prior (float) : float value of a prior possibility for being negative (logarithmic scale)
    """
    
    data_path = os.listdir(dirname)
    
    for file in data_path:
        filepath = os.path.join(dirname, file)
        positive = positive_prior
        negative = negative_prior
        
        with open(filepath) as text_file:
            for line in text_file:
                words = clean_text(line, interpunctions, stopwords)
                for word in words:
                    if word in positive_probabilites:  # Checks if the word is in our trained data, else skips 
                        positive += positive_probabilites[word]
                        negative += negative_probabilities[word]
            if positive > negative:
                print("the file is positive")
            else:
                print("The file is negative")
                        

def main():
    interpunctions = read_file(INTERPUNCTIONS)
    stopwords = read_file(REMOVE_WORDS)
    
    positive_probabilities, negative_probabilities, positive_prior, negative_prior = create_classification(
        TRAININGPATH, interpunctions, stopwords)
    
    classify(TESTPATH, interpunctions, stopwords, positive_probabilities,
             negative_probabilities, positive_prior, negative_prior)


if __name__ == "__main__":
    main()

