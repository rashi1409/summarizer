from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request

from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
from sumbasic_summ import main
from textrank import textranksumm
import time
import spacy
import re
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text
#summary length
def sumlen(summ):
	res = len(re.findall(r'\w+', summ)) 
	return res
@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = main(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)

@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = main(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		#time_saved=final_reading_time-summary_reading_time
		len_summ=sumlen(final_summary_spacy)
		#Gensim Summarizer
		final_summary_gensim = summarize(rawtext)
		summary_reading_time_gensim = readingTime(final_summary_gensim)
		#time_saved_gensim=final_reading_time-summary_reading_time_gensim
		len_gensim=sumlen(final_summary_gensim)
        
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		summary_reading_time_nltk = readingTime(final_summary_nltk)
		#time_saved_nltk=final_reading_time-summary_reading_time_nltk
		len_nltk=sumlen(final_summary_nltk)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy) 
		#time_saved_sumy=final_reading_time-summary_reading_time_sumy
		len_sumy=sumlen(final_summary_sumy)
		#sumbasic
		final_summary_sumbasic=main(rawtext)
		summary_reading_time_sumbasic=readingTime(final_summary_sumbasic)
		#time_saved_sumbasic=final_reading_time-summary_reading_time_sumbasic
		len_sumbasic=sumlen(final_summary_sumbasic)
		#textrank
		final_summary_textrank=textranksumm(rawtext) 
		summary_reading_time_textrank=readingTime(final_summary_textrank)
		#time_saved_textrank=final_reading_time-summary_reading_time_textrank
		len_textrank=sumlen(final_summary_textrank)
        
		end = time.time()
		final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_nltk=summary_reading_time_nltk,final_summary_sumbasic=final_summary_sumbasic,summary_reading_time_sumbasic=summary_reading_time_sumbasic,final_summary_textrank=final_summary_textrank,summary_reading_time_textrank=summary_reading_time_textrank,len_summ=len_summ,len_nltk=len_nltk,len_sumbasic=len_sumbasic,len_textrank=len_textrank)



@app.route('/about')
def about():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
