import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from bs4 import BeautifulSoup
import requests
import re

import nltk


def _input(topic):
	article = ""
	link = "https://en.wikipedia.org/wiki/" + topic.strip() 
	page = requests.get(link)
	content = BeautifulSoup(page.content,'html.parser')
	paragraphs = content.find_all('p')
	for paragraph in paragraphs:
		article+= paragraph.text+" "
	return article

def clean(sentences):
	lemmatizer = WordNetLemmatizer()
	cleaned_sentences = []
	for sentence in sentences:
		sentence = sentence.lower()
		sentence = re.sub(r'[^a-zA-Z]',' ',sentence)
		sentence = sentence.split()
		sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in set(stopwords.words('english'))]
		sentence = ' '.join(sentence)
		cleaned_sentences.append(sentence)
	return cleaned_sentences

def init_probability(sentences):
	probability_dict = {}
	words = word_tokenize('. '.join(sentences))
	total_words = len(set(words))
	for word in words:
		if word!='.':
			if not probability_dict.get(word):
				probability_dict[word] = 1
			else:
				probability_dict[word] += 1

	for word,count in probability_dict.items():
		probability_dict[word] = count/total_words 
	
	return probability_dict

def update_probability(probability_dict,word):
	if probability_dict.get(word):
		probability_dict[word] = probability_dict[word]**2
	return probability_dict

def average_sentence_weights(sentences,probability_dict):
	sentence_weights = {}
	for index,sentence in enumerate(sentences):
		if len(sentence) != 0:
			average_proba = sum([probability_dict[word] for word in sentence if word in probability_dict.keys()])
			average_proba /= len(sentence)
			sentence_weights[index] = average_proba 
	return sentence_weights

def generate_summary(sentence_weights,probability_dict,cleaned_article,tokenized_article,summary_length = 30):
	summary = ""
	current_length = 0
	while current_length < summary_length :
		highest_probability_word = max(probability_dict,key=probability_dict.get)
		sentences_with_max_word= [index for index,sentence in enumerate(cleaned_article) if highest_probability_word in set(word_tokenize(sentence))]
		sentence_list = sorted([[index,sentence_weights[index]] for index in sentences_with_max_word],key=lambda x:x[1],reverse=True)
		summary += tokenized_article[sentence_list[0][0]] + "\n"
		for word in word_tokenize(cleaned_article[sentence_list[0][0]]):
			probability_dict = update_probability(probability_dict,word)
		current_length+=1
	return summary

def main(rawtext):
	#topic = input("Enter the title of the wikipedia article to be scraped----->")
	article = rawtext
	required_length = 5
	tokenized_article = sent_tokenize(article)
	cleaned_article = clean(tokenized_article) 
	probability_dict = init_probability(cleaned_article)
	sentence_weights = average_sentence_weights(cleaned_article,probability_dict)
	summary = generate_summary(sentence_weights,probability_dict,cleaned_article,tokenized_article,required_length)
	return summary

#if __name__ == "__main__":
	#main()

