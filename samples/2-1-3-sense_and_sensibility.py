import PyPDF2
import os 
import string
import time

'''
Please note that in creating this dictionary, we will only document the usage 
frequency of English words in the novel. For example, we shall ignore some 
patterns: 
(1) Numerical numbers such as 1, 2, 3 
(2) Punctuation and space 
(3) Headlines such as “CHAPTER 1” 
(4) Page numbers such as “page 1 / 559” 
Hint: use string.split(‘slash n’) to split a text string to lines; use string.split() to split 
a text string to words.
'''

path = os.path.dirname(os.path.abspath(__file__))
file_handle = open('Sense-and-Sensibility-by-Jane-Austen.pdf', 'rb') 
pdfReader = PyPDF2.PdfReader(file_handle) 

frequency_table = dict()

start_time = time.time()

for i in range(len(pdfReader.pages)):   # this tells you total pages 
    page_object = pdfReader.pages[i]    # We just get page 0 as example 
    text = page_object.extract_text()   # this is the str type of full page
    #all pages and then all lines on each page
    for i in text.split(" "):
        if i.isalpha() and not i=="CHAPTER": # theres no instances of "page 1/569"
            i = i.translate(str.maketrans('', '', string.punctuation))
            if (i.lower() not in frequency_table):
                frequency_table[i.lower()]=1
            else:
                frequency_table[i.lower()]+=1

end_time = time.time()
file_handle.close()

total_distinct_words = len(frequency_table)
print("ttotal number of unique words: " + str(total_distinct_words))
print("time taken " + str(end_time-start_time))
print(frequency_table)