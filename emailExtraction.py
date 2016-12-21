# Text Mining - Hillary Clinton Emails 
import urllib
import json
import gensim
import nltk
import re
import string
import time
import csv


# https://www.leaksapi.com/profile

key='secret_key'
WIKILEAKS_URL = 'https://www.leaksapi.com/clinton-emails/id/'
# Initialize empty arrays
emaillist = []
splitList = []

# All other times you run you ARE REQUIRED TO CREATE AN EMPTY LIST
#emaillist = [None]*25741
#splitList = [None]*25741

# Force utf-8 encoding on characters that do not meet this criteria
def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


def callAPI():
	# Create CSV file with header
	with open( 'c:/Users/Sean Ankenbruck/Desktop/MSA/TextMining/emails.csv', 'wb' ) as fp:
		out = csv.writer( fp )
		out.writerow(["EmailID","Unclassified","From","Time","To","Body"])
		for i in range( 30,322 ):
			email = i
			# Append our query string onto the end of the URL request (with the required params (sensor and address))
			url = WIKILEAKS_URL + key + '/' + str(email)
			# Open URL
			uh = urllib.urlopen(url)
			data = uh.read()
			# Parse the json
			try: js = json.loads(str(data))
			except: js = None
			# Remove unicode encoding
			body = byteify(js)
			# If email index == 0, skip it (index zero throws an error)
			if i == 0:
				continue
			# Else append the body to emaillist
			else:
				emaillist.append(body)
			# Find special characters of type \n or \t (or multiple of each) and replace with a space
			rmvChar = re.sub( '[\n+\t+]', ' ', emaillist[i]['body'] )
			# Then find spaces (or multiple spaces) and replace with one space
			rmvSpace = re.sub( ' +', ' ', rmvChar )
			# Now we will use this new "clean" format to parse specific pieces for our analysis
			# Test for UNCLASSIFIED = Email signature
			if 'UNCLASSIFIED' in rmvSpace:
				unclassified=re.findall(r'UNCLASSIFIED(.*?)From:',rmvSpace)
			else:
				unclassified='NOT FOUND'
			# Test for from
			if 'From:' in rmvSpace:
				# FROM = Who sent the message
				fromWho=re.findall(r'From:(.*?)Sent:',rmvSpace)
			else:
				fromWho='NOT FOUND'
			# Test for Time Sent
			if 'Sent:' in rmvSpace:
				# SENTTIME = Time message was sent
				sentTime=re.findall(r'Sent:(.*?)To:',rmvSpace)
			else:
				sentTime='NOT FOUND'
			# Test for Subject
			if 'Subject:' in rmvSpace:
				# SENTTO = Who received the message
				sentTo=re.findall(r'To:(.*?)Subject:',rmvSpace)
				seperate='Subject:'
				target=rmvSpace.split(seperate, 1)[1]
			else:
				sentTo="Not found"
				seperate='UNCLASSIFIED'
				target = rmvSpace.split(seperate, 1)[1]
			# Use rsplit() to keep only original messages, get rid of reply thread
			emailBody = target.rsplit('Original Message', 1)[0]
			#print emailBody
			#status: SUCCESS
			out.writerow([i]+[unclassified]+[fromWho]+[sentTime]+[sentTo]+[emailBody])
			print i
			i = i + 1
			# Prevents the API from rate limiting our requests
			time.sleep(2)


callAPI()
# Test at 1 email: SUCCESS
# Test at 100 email: SUCCESS
# Extraction successful - time elapsed: 16+ hours



