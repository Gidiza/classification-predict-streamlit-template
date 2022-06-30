"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
#%matplotlib inline
# Vectorizer
news_vectorizer = open("resources/vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/apptrain.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	cola, mid, colb = st.columns([40,1,40])
	with cola:
		st.image('logo2.png', width=200)
	with colb:
		st.title("JHB-Intelect data")
	
	st.title("")
	st.text("")
	st.text("")


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	with st.sidebar:
		selection = option_menu("Main Menu", ["Home", 'visualisation', 'Development team','Contact Us'], 
        icons=['house', 'pie-chart', 'people-fill', 'envelope'], menu_icon="cast", default_index=1)
	#options = ["Prediction", "Information", "Development Team","Information 2" ]
	#selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information 2":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiments', 'message']]) # will write the df to the page
			st.write('Show sentiment occurance dataset')
			xx = raw['sentiments'].value_counts()
			st.bar_chart(xx)

	# Building out the predication page
	if selection == "Home":
		col1, mid, col2 = st.columns([40,1,20])
		with col1:
			st.subheader("Tweet classifier for climate change tweet classification")
		with col2:
			st.image('twitter.webp', width=80)

		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/mlr_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			prediction_dic =  {-1:"Anti: the tweet does not believe in man-made climate change", 0:"Neutral: the tweet neither supports nor refutes the belief of man-made climate change",
			1:"Pro: the tweet supports the belief of man-made climate change", 2:"News: the tweet links to factual news about climate change"}
			st.success("Text Categorized as: {}".format(prediction_dic[prediction[0]]))
	if selection == "Development team":
		st.title("Meet our team")
		st.title("")
		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Gwetha Nkosi - Project Manager")
			st.write("Gcwetha Nkosi has worked as a Project Manager, Product Manager, Systems and Production developer. When he is not coding he enjoys watching sport on television.")
		with col1:
			st.image('nkosi2.jpeg', width=380)
		col1, mid, col2 = st.columns([80,10,80])
		with col1:
			st.subheader("Jowellyn Afrikaner - Data Scienstist")
			st.write("Jowellyn has worked as a data scientist for various companies including Netflix and Apple to name a few. In his spare time he likes to spend time with family and watch football")
		with col2:
			st.image('jowellyn2.jpeg', width=380)		


		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Nolo Masetlwa- Machine learning engineer")
			st.write("She has designed predicted models for companies such as FNB and BMW. One of my project was creating a chatbot with Python's NTLK library.She is a fitness fanatic and loves dancing ")

		with col1:
			st.image('nolo.jpeg', width=380)
		
		col1, mid, col2 = st.columns([80,10,80])
		with col1:
			st.subheader("Nelisiwe Mathebula- Data Analyst")
			st.write("Nelisiwe is a data Analyst Intern, participated on four of our outstanding projects thus far. She enjoys indoor atmosphere and on her spare time she plays puzzle video games")

		with col2:
			st.image('nelly2.jpeg', width=380)

		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Ally monareng - Data scientists")
			st.write("Ally is an expiriences Data scientists he has been working in the field for ten years, he worked worked for several companies , he is also the founder of the 'ALLOSINA  TEXTILE AND DESIGNS', he loves taking part on kaggle competitions")

		with col1:
			st.image('ally.jpeg', width=380)

		col1, mid, col2 = st.columns([80,10,80])
		with col1:
			st.subheader("Mtshali Lindokuhle - App developer")
			st.write("Lindokuhle has worked as an App developer on multiple project with different companies like Paypal, showmax just to name a few. On his spare time he likes watching movies and playing video games")
		with col2:
			st.image('me.jpeg', width=380)


		

	if selection == "visualisation":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			
			st.write(raw[['sentiments', 'message']]) # will write the df to the page

			opt = st.radio('Plot  type:',['Bar', 'Pie', 'Word Cloud'])
			if opt=='Bar':
				st.markdown('<h3>Show sentiment occurance dataset</h3>',unsafe_allow_html=True)
				xx = raw['sentiments'].value_counts()
				st.bar_chart(xx)
			elif opt =="Pie":
				st.markdown('<h3>Pie chart for percentage of each sentiment on dataset</h3>',unsafe_allow_html=True)
				fig1, ax1 = plt.subplots()
				ax1.pie(raw['sentiments'].value_counts(),labels = ["Pro","News","Neutral","Anti"], autopct='%1.1f%%',shadow=True, startangle=90)
				ax1.axis('equal')
				ax1.set_facecolor("black")  # Equal aspect ratio ensures that pie is drawn as a circle.
				ax1.legend()
				fig1.patch.set_alpha(0)
				ax1.xaxis.label.set_color('red')
				st.pyplot(fig1)
				
		
			else:
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.markdown('<h3>Word Cloud for how frequently words show up on all tweets.</h3>',unsafe_allow_html=True)
				allwords = ' '.join([msg for msg in raw['message']])
				WordCloudtest = WordCloud(width = 800, height=500, random_state = 21 , max_font_size =119).generate(allwords)
				
				plt.imshow(WordCloudtest, interpolation = 'bilinear')
				
				plt.axis('off')
				st.pyplot(plt.show())
	if selection =="Contact Us":
		col1, mid, col2 = st.columns([80,1,80])
		with col1:
			st.subheader("Contact Us")
		with col2:
			st.image('c22.png', width =80)
		st.write("email: jhb-data-intalect@gmail.com")
		st.write("phone: (+27) 23 456 789")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
