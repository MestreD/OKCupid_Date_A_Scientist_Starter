#Libraries to work with
## General libraries
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
#This will change the renderer to a version that uses the Plotly JS code directly and in online mode.
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
plt.rcParams['figure.figsize'] = [6, 6]
import streamlit as st
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

# MAchine learning library to split train and test, classification models and evaluating scores. 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Functions to work with in this project:
class all_df_balanced():
    def __init__(self, df, percentage, class_drinks, total_class_drinks, cols):
        self.df = df
        self.percentage = percentage
        self.class_drinks = class_drinks
        self.total_class_drinks = total_class_drinks
        self.cols = cols
# Labels Encoding
    def encoder(df, cols):
        le_list = []
        le_tr = []
        df_encoded = pd.DataFrame()
        for i in range(len(cols)):
            le_list.append(preprocessing.LabelEncoder())
            le_tr.append(le_list[i].fit_transform(df[cols[i]]))
            df_encoded[cols[i]] = le_tr[i]
        return df_encoded
# Label Balance
    def index_random(df, percentage, class_drinks):
        df = all_df_balanced.encoder(df, cols)
        series_targets = df["drinks"].value_counts(normalize=True) 
        class_drinks_percentage = series_targets[class_drinks]
        df = df.loc[df.drinks == class_drinks]
        list_indx = df.index.values.tolist()
        np.random.seed(1)
        new_indx = np.random.choice(list_indx, replace=True, size = int(len(df)*percentage/class_drinks_percentage))
        df_out = df.loc[new_indx]
        return df_out
    
    def all_dfs(df, percentage, total_class_drinks):
        list_df = []
        for i in range(0,total_class_drinks,1):
            class_drinks = i
            list_df.append(all_df_balanced.index_random(df, percentage, class_drinks))
        df_balanced = pd.concat(list_df, join="inner")
        df_out = df_balanced.sample(frac = 1)
        #the labels are the target to test the features with.
        features = df_out.iloc[:,0:-1]
        drinks_labels = df_out.iloc[:,-1:]

        #Split the data into chunks
        X_train, X_test, y_train, y_test = train_test_split(features, drinks_labels, test_size=0.25, random_state = 42)
        models = [LogisticRegression(multi_class="multinomial"), KNeighborsClassifier(), RandomForestClassifier()]

        # Model building
        plt.figure(figsize=(25, 7))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("Confusion Matrix", fontsize=18, y=0.95)
        for n, model in enumerate(models):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            model_labels = ['socially', 'often', 'not at all', 'rarely', 'very often',
       'desperately']
            ax = plt.subplot(1, 3, n + 1)
            sns.heatmap(cm, annot=True, ax = ax, fmt="d");
            # labels, title and ticks
            ax.set_xlabel('Predicted labels');
            ax.set_ylabel('True labels'); 
            ax.set_title([model, "F1 Score= ",f1_score(y_test, y_pred, average='weighted')]);
            ax.yaxis.set_tick_params(rotation=360)
            ax.xaxis.set_tick_params(rotation=90)
            ax.xaxis.set_ticklabels(model_labels);
            ax.yaxis.set_ticklabels(model_labels);
        plt.show()
        
        

# Streamlit web.
st.set_page_config(page_title="Ok Cupid Date a Scientist",
        page_icon=("❤️"), layout="wide")

profiles = pd.read_csv('profiles.csv')

header = st.container()
dataset = st.container()
features = st.container()

with header:
    st.title("Ok Cupid Date a Scientist.")
    col1, col_mid, col2 = st.columns((1, 0.1, 1))
    with col1:   
        st.write("This project analyzes data from on-line dating application OKCupid. In recent years, there has been a massive rise in the usage of dating apps to find love. Many of these apps use sophisticated data science techniques to recommend possible matches to users and to optimize the user experience. These apps give us access to a wealth of information that we've never had before about how different people experience romance. The goal of this project is to scope, prep, analyze, and create a machine learning model to solve a question.")
    with col2:
        image = Image.open('okcupid.png')
        st.image(image, width=400)
    st.markdown("""---""") 
    st.markdown("### Project Goals")
    st.write("In this project, the goal is to apply machine learning techniques to a data set. The primary research question that will be answered is whether an OkCupid's drinking habit can be predicted using other variables from their profiles. This project is important since sharing a lifestyle and habits can be important part of matches, and if users don't input their drinking habit, OkCupid would like to predict which habit they might be. ")
    st.markdown("### Data")
    st.write("The project has one data set provided by Codecademy called  \"profiles.csv\". In the data, each row represents an OkCupid user and the columns are the responses to their user profiles which include multi-choice and short answer questions.")
    st.markdown("###  Analysis")
    st.write("This solution will use descriptive statistics and data visualization to find key figures in understanding the distribution, count, and relationship between variables. Since the goal of the project to make predictions on the user's drinking habits, classification algorithms from the supervised learning family of machine learning models will be implemented. ")
    st.markdown("### Evaluation")
    st.write("The project will conclude with the evaluation of the machine learning model selected with a validation data set. The output of the predictions can be checked through a confusion matrix, and metrics such as accuracy, precision, recall, F1 and Kappa scores.")
with dataset:
    st.markdown("#### Data Characteristics")
    st.write("**profiles** has 59,946 rows and 31 columns, this is a good sign since there seems to be enough data for machine learning. ")
    col1, col_mid, col2 = st.columns((1, 0.1, 1))
    with col1:
        st.markdown("The columns in the dataset include:")
        st.markdown("""
    - **age:** continuous variable of age of user \n
    - **body_type:** categorical variable of body type of user\n
    - **diet:** categorical variable of dietary information\n
    - **drinks:**  categorical variable of alcohol consumption\n
    - **drugs:** categorical variable of drug usage\n
    - **education:** categorical variable of educational attainment\n
    - **ethnicity:** categorical variable of ethnic backgrounds\n
    - **height:** continuous variable of height of user\n
    - **income:** continuous variable of income of user\n
    - **job:** categorical variable of employment description\n
    - **offspring:** categorical variable of children status\n
    - **orientation:** categorical variable of sexual orientation\n
    - **pets:** categorical variable of pet preferences\n
    - **religion:** categorical variable of religious background\n
    - **sex:** categorical variable of gender\n
 """)
    with col2:
        st.markdown("""
    - **sign:** categorical variable of astrological symbol\n
    - **smokes:** categorical variable of smoking consumption\n
    - **speaks:** categorical variable of language spoken\n
    - **status:** categorical variable of relationship status\n
    - **last_online:** date variable of last login\n
    - **location:** categorical variable of user locations""")
        st.markdown("And a set of open short-answer responses to :")
        st.markdown("""
    - **essay0:** My self summary\n
    - **essay1:**  What I’m doing with my life\n
    - **essay2:** I’m really good at\n
    - **essay3:** The first thing people usually notice about me\n
    - **essay4:** Favorite books, movies, show, music, and food\n
    - **essay5:** The six things I could never do without\n
    - **essay6:** I spend a lot of time thinking about\n
    - **essay7:** On a typical Friday night I am\n
    - **essay8:** The most private thing I am willing to admit\n
    - **essay9:** You should message me if…""")
st.markdown("""---""") 
st.markdown("## Explore the Data")
st.write("In order to get insight from the data and after knowing the missing values that could be valuable for the user and therefore for the company, I think that a good idea could be predicting the drink habits to suggesting the user to choose from our prediction.  Lets explore those features to see if I can work with those values.")
st.write("The number of categories:",profiles.drinks.nunique())
st.write("Categories:", profiles.drinks.unique())  
st.markdown("""---""") 
st.markdown("## Clean Labels")
st.write("It is important that we clean the labels since this is what will be predicted and 48 predictions would be quite difficult. By taking the first word of the column, the signs and religion can be saved without the qualifiers. The qualifiers could be used for another problem down the line.")
profiles["sign"] = profiles["sign"].str.split().str.get(0)
profiles["religion"] = profiles.religion.str.split().str.get(0)
st.markdown("""---""") 
st.markdown("## Continuos Variables")
st.markdown("#### Age")
st.write("The next plot shows the distribution of age in the group. It seems that most users are in their late 20s to early 30s. And that there are proportionally similar break down of gender by age, but slightly fewer females overall." )
fig = px.histogram(profiles, x="age", color="sex", barmode="overlay")
st.plotly_chart(fig)
st.markdown("#### Income")
st.write("Here is the data of income, it seems that the majority of the participants do not include their income figures. Maybe because they don't want this to be a decision making for th possible matches.")
fig = px.histogram(profiles, x="income", color="sex", barmode="overlay")
st.plotly_chart(fig)
st.markdown("### Discrete Variables")
st.markdown("#### Body Type")
st.write("The next chart shows the body type variable, and it seems that most users will describe themselves as average, fit, or athletic.")
body_count = profiles.groupby(by=["body_type"]).size().reset_index(name="counts")
fig = px.bar(data_frame=body_count, x="counts", y="body_type", barmode="group", color="body_type", color_discrete_sequence=px.colors.qualitative.Antique)
fig.update_traces(width=1)
st.plotly_chart(fig)
st.markdown("#### Diet")
st.write("Here is a chart of the dietary information for users. Most user eat \"mostly anything\", followed by \"anything\", and \"strictly anything\", being open-minded seems to be a popular signal to potential partners.")
diet_count = profiles.groupby(by=["diet"]).size().reset_index(name="counts")
st.plotly_chart(px.bar(data_frame=diet_count, x="counts", y="diet", color="counts", barmode="group"))
st.markdown("#### Drinks")
st.write("The next plot shows that the majority of the users drink \"socially\", then \"rarely\" and \"often.\"")
drinks_count = profiles.groupby(by=["drinks"]).size().reset_index(name="counts")
st.plotly_chart(px.bar(data_frame=drinks_count, x="counts", y="drinks", color="counts", barmode="group"))
st.markdown("#### Drugs")
st.write("The vast majority of users said \"never\" use drugs. ")
drugs_count = profiles.groupby(by=["drugs"]).size().reset_index(name="counts")
st.plotly_chart(px.bar(data_frame=drugs_count, x="counts", y="drugs", color="counts", barmode="group"))
st.markdown("#### Education")
st.write("Below you can see the majority of users are graduate from college/university followed by masters programs and those working on college/university. Interestingly space camp related options are fairly a popular options.")
education_count = profiles.groupby(by=["education"]).size().reset_index(name="counts")
fig = px.bar(data_frame=education_count, x="counts", y="education", color="education", barmode="group")
fig.update_traces(width=1)
st.plotly_chart(fig)
st.markdown("#### Jobs")
st.write("Most users don't fit into the categories provided, but there are a fair share of students, artists, tech, and business folks.")
jobs_count = profiles.groupby(by=["job"]).size().reset_index(name="counts")
fig = px.bar(data_frame=jobs_count, x="counts", y="job", color="job", barmode="group")
fig.update_traces(width=1)
st.plotly_chart(fig)
st.markdown("#### Orientation")
st.write("The majority of users are straight. Interestingly the majority of bisexual users are female. ")
orientation_sex_count = profiles.groupby(by=["orientation", "sex"]).size().reset_index(name="counts")
st.plotly_chart(px.bar(data_frame=orientation_sex_count, x="counts", y="orientation", color="sex", barmode="group"))
st.markdown("#### Religion")
st.write("Religion was similar to sign where there are a lot of qualifiers and was cleaned to take the first word and distilled down to 9 groups. The majority was not very religious identifying as agnostic, other, or atheists.") 
religion_count = profiles.groupby(by=["religion"]).size().reset_index(name="counts")
fig = px.bar(data_frame=religion_count, x="counts", y="religion", color="religion", barmode="group")
fig.update_traces(width=1)
st.plotly_chart(fig)
st.markdown("#### Smoking")
st.write("Similarly for drugs the majority of users chose \"no\" for smoking.")
smoking_count = profiles.groupby(by="smokes").size().reset_index(name="counts")
st.plotly_chart(px.bar(data_frame=smoking_count, x="counts", y="smokes", color="smokes"))



col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:    
        st.header("Data Acquisition")  
        st.write("In this project I will use the data stored in  YahooFinance Website with Yahoo! Finance's API")
        st.write('[Documentation](http://theautomatic.net/yahoo_fin-documentation/)')   
with col2:
#displaying the image on streamlit app
        st.image(image)
        st.write()  
st.markdown("""---""") 

