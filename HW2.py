''' This program contains 5 methods:
    The read_data(fileName) reads the data into a list of lines.
    The clean_data(my_data) cleans the data using regex and returns a list of cleaned data
    The build_n_gram_dict(n, cleaned_data) takes the cleaned data and builds a specified n-gram with all words in the cleaned_data
    The calculate_PP(test_sentences, ngram_models) uses the dictionary of dictionaries of n-grams to calculate how correct our language model is based off of test_sentences
    The generate_text(ngram_models, text_length, seed_word) uses the dictionary of dictionaries to generate text based off of the text_length using the seed_word.
'''

__author__ = "Manoj Munaganuru"
__version__="03.08.19"
import re
import random

def read_data(fileName = ""):
    if type(fileName) is not str or fileName == "": #check whether the parameter passed in as a string
        return []
    try:
        someFile = open(fileName, "r") #open file and read lines
        lines = someFile.readlines()
        for i in range(len(lines)):
            lines[i] = re.sub("\n", "", lines[i]) #split file based off of newline character
        return lines
    except IOError:
        return [] #if ioerror occurs, return empty list

def clean_data(my_data = []):
    cleaned_data = []
    if type(my_data) is not list or my_data == []:
        return []
    for line in my_data:
        line = line.lower() #make lines lower case
        line = re.sub("--", " ", line) #any double dash are turned into space, single dashes are turned into empty string
        line = re.sub("-", "", line)
        line = re.sub("[^-1-9a-zA-Z']", " ", line) #takes out all non alpha numeric characters besides aposterphes
        line = re.sub(" +", " ", line) #gets rid of any leading or trailing whitespace
        words = line.split()
        for i in range(0, len(words)): #check whether words begin with numbers and puts spaces between them
            someRegex = re.search(r'\d+$', words[i])
            if words[i].isdigit():
                continue
            if someRegex is not None and words[i].startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                words[i] = re.sub(r'(-?[0-9]+\.?[0-9]*)', r" \1", words[i])
                words[i] = re.sub(r'(-?[0-9]+\.?[0-9]*)', r"\1 ", words[i])
            if someRegex is not None:
                words[i] = re.sub(r'(-?[0-9]+\.?[0-9]*)', r" \1", words[i])
            if words[i].startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                words[i] = re.sub(r'(-?[0-9]+\.?[0-9]*)', r"\1 ", words[i])
        temp = ""
        for i in range(0, len(words) - 1):
            temp += words[i] + " "
        temp = temp + words[len(words) - 1]
        line = temp
        temp = ""
        words = line.split()
        for i in range(0, len(words) - 1): #if the word is a number, it gets substituted with num
            if words[i].isdigit():
                words[i] = "num"
            temp += words[i] + " "
        if words[len(words) - 1].isdigit():
            words[len(words) - 1] = "num"
        temp = temp + words[len(words) - 1]
        line = temp
        words = line.split()
        i = 0
        while i < len(words) - 1: #if the word and word next to it is num, then the second one gets removed
            if words[i] == "num" and words[i + 1] == "num":
                del(words[i + 1])
                i -= 1
            i += 1
        temp = ""
        for i in range(0, len(words) - 1):
            temp += words[i] + " "
        temp = temp + words[len(words) - 1]
        line = temp
        cleaned_data.append(line) #append this line into the cleaned_data list
    return cleaned_data

def build_n_gram_dict(n = 0, cleaned_data = []):
    if type(n) is not int or type(cleaned_data) is not list or n == 0 or cleaned_data == []: #check for extraneous parameters that will result in error
        return {}
    ngram_dict = {}
    for i in range(0, len(cleaned_data)): #put sentence tag before all cleaned_dta
        cleaned_data[i] = "<s> " + cleaned_data[i]
    for i in range(0, len(cleaned_data)):
        words = cleaned_data[i].split()
        for j in range(0, len(words) - n + 1): #gets n number of words in a specific string
            res = ""
            if (j + n) > len(words):
                break
            for k in range(0, n - 1):
                res += words[j + k] + " "
            res += words[j + n - 1]
            if res == "<s>":
                continue
            if res in ngram_dict: #if key already exists in dict, then add 1 ; else make new dict entry
                ngram_dict[res] += 1
            else:
                ngram_dict[res] = 1
    return ngram_dict

def calculate_PP(test_sentences = [], ngram_models = {}):
    if type(test_sentences) is not list or type(ngram_models) is not dict or test_sentences == [] or ngram_models == {}:
        return 0
    n = len(ngram_models)
    list_dict = []
    for i in range(1, n + 1):
        list_dict.append(ngram_models[i])
    V = len(ngram_models[1])
    perplexities = []
    for i in range(0, len(test_sentences)):
        strings = test_sentences[i].split()
        res = ""
        res_small = ""
        prob = 1.0
        k = 1
        while k + 1 < n: #deals with edge cases of calculating perplexity when number of words is less than number of n-grams in ngram models
            for l in range(0, k):
                res = res + strings[l] + " "
            res = res + strings[k]
            for l in range(0, k - 1):
                res_small += strings[l] + " "
            res_small = res_small + strings[k - 1]
            try:
                prob *= float((ngram_models[k + 1][res] + 1))/float((ngram_models[k][res_small] + V))
            except KeyError: #calculate perplexity and if key is never found, let probability equal 0
                prob *= 0
            res = ""
            res_small = ""
            k += 1
        for m in range(k, len(strings)): #when you get to part of sentence, where there is enough context to use max ngram dict
            for l in range(m - k, m):
                res = res + strings[l] + " "
            res = res + strings[m]
            for l in range(m - k, m - 1):
                res_small += strings[l] + " "
            res_small = res_small + strings[m - 1]
            try:
                prob *= float((ngram_models[n][res] + 1))/float((ngram_models[n-1][res_small] + V)) #calculate perplexity and if key is never found, let probability equal 0
            except KeyError:
                prob *= 0
            res = ""
            res_small = ""
        if prob == 0:
            perplexities.append(0) #if the prob is 0, perplexity is also 0: did this because python gives error if you do 0 to power of anything
        else:
            perplexities.append((prob ** (-1 / len(strings)))/len(test_sentences)) #else, append the perplexity using formula that was in handout
    total = 0
    for i in range(0, len(perplexities)): #use loop to go through individual perplexities and calculate average perplexity
        total += perplexities[i]
    return total/len(perplexities)

def generate_text(ngram_models = {}, text_length = 0, seed_word = ""):
    if type(ngram_models) is not dict or type(text_length) is not int or type(seed_word) is not str or ngram_models == [] or text_length == 0 or seed_word == "": #check for extraneous parameters
        return ""
    result = seed_word
    one_gram = ngram_models[1].keys() #get all unique words in language model
    V = len(one_gram)
    n = len(ngram_models)
    for i in range(1, text_length): #go through all text-length
        small_res = ""
        word_choice = ""
        result1 = result.split()
        prob = -1
        if i < n:
            for j in range(i):
                small_res += result1[j] + " "
            small_res = small_res.strip() #get small string for calculating perplexity
        else:
            for j in range(i - n + 1, i):
                small_res += result1[j] + " "
            small_res = small_res.strip()
        for k in range(V):
            big_res = small_res + " " + one_gram[k]
            lmao = big_res.split()
            try:
                temp = float(ngram_models[len(lmao)][big_res] + 1)/float(ngram_models[len(lmao) - 1][small_res] + V) #use this to calculate probability
            except KeyError:
                temp = 0
            if temp >= prob: #check which prob is the greatest, and use that word
                prob = temp
                word_choice = one_gram[k]
        if prob == 0:
            word_choice = one_gram[random.randint(0, len(one_gram))]
        result += " " + word_choice #concatenate the word choice to the result
    return result


def main():
    word = raw_input("Please enter a seed word: ")
    length = raw_input("How many words would you like to generate? ")
    print generate_text({1: build_n_gram_dict(1, clean_data(read_data("text8"))),
                         2: build_n_gram_dict(2, clean_data(read_data("text8"))),
                         3: build_n_gram_dict(3, clean_data(read_data("text8")))}, int(length), word)

main()
