import PyPDF2
import os 
import string
import time

start_time = time.time()

path = os.path.dirname(os.path.abspath(__file__))
file_handle = open('Sense-and-Sensibility-by-Jane-Austen.pdf', 'rb') 
pdfReader = PyPDF2.PdfReader(file_handle) 
page_number = len(pdfReader.pages)   # this tells you total pages 
page_object = pdfReader.pages[0]    # We just get page 0 as example 
page_text = page_object.extractText()   # this is the str type of full page

frequency_table = dict()

#iterating through the pages

for page_num in range(len(pdfReader.pages)):
    page_object = pdfReader.pages[page_num]
    page_text = page_object.extract_text()
    words = page_text.split("\n")
    for extracted_string in words:
        #clean the selected word
        extracted_string2 = extracted_string.split("\n")
        word = word.strip(string.punctuation).lower()

    if word.isalpha():
        if word in frequency_table:
            frequency_table[word] +=1
        else:
            frequency_table[word] = 1


end_time = time.time()
fild_handle.close()

total_distinct_words = len(frequency_table)
print("ttotal number of unique words: " + str(total_distinct_words))
print("time taken " + str(end_time-start_time))