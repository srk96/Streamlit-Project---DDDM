import streamlit as st
import time
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from PIL import Image 
from openpyxl.workbook import Workbook
import locale
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from timeit import default_timer as timer
from math import isnan
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import category_encoders as ce
import warnings
warnings.filterwarnings('ignore')

# Adding %-formatting
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

#display a wider page 
st.set_page_config(layout="wide")

#implement a function to run the code faster 
@st.cache(allow_output_mutation=True)
def get_data(df):
 df=st.write(df)
 return df
 df=get_data(df)


#set the background color
st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
  )

background_color= '#F5F5F5'

#add some infos
b1, b2 = st.columns([5,1])
with b1:
    st.text("")
    st.write('Made by _**Stephanie Khabbaz**_')
    st.write('**MSBA AUB**')
    st.markdown("<a style='text-align: center;' href= 'https://www.linkedin.com/in/stephanie-khabbaz/'>LinkedIn</a>", unsafe_allow_html=True)
with b2:
    st.image('Logo MSBA.png', width=150)

st.text("")

# as horizontal menu
selected=option_menu(menu_title= None, #required
	    options=['Home', 'Data Exploration', 'Descriptive Analytics','Predictive Analytics', 'Recommendations'], #required
		icons=["house-door", "binoculars", "file-earmark-bar-graph","graph-up-arrow", "lightbulb"], #optional
		default_index=0, #optional 
		orientation='horizontal',
		styles={"container": {"background-color": "#DCDCDC	"}, "icon": {"color": "black", "font-size": "20px"}, "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color":"#FFFFFF"},
		"nav-link-selected":{"background-color":"#4682B4"},})

st.text("")
st.text("")

image = Image.open('global superstore.png')

if selected == "Home":

  input_col,picture_col=st.columns([0.5, 1.5])
  picture_col.image(image, use_column_width=None, output_format='auto', width= 500, clamp=[255])

  st.markdown("<h1 style='text-align: center; color: #4682B4'><b/>Global Superstore Sales Analysis<b/></h1>", unsafe_allow_html=True)

  st.text("")
  
  
  phrase= '<p style="color:black; font-size: 16px; text-align: center"><b/><i/>Where seamless decision-making is created through data visualization.<b/><i/></p>'
  st.markdown(phrase, unsafe_allow_html=True)

  st.text("")
  st.text("")

  st.text("")

  st.text("")
  st.text("")

  picture_col_1, picture_col_2, picture_col_3= st.columns([0.5, 0.5, 0.2])
  picture_col_1.image('data visualization.png', width=180)
  picture_col_2.image('data viz.png', width=180)
  picture_col_3.image('data exploration.png', width=180)
  st.text("")
   
  


  st.text("")
  st.text("")
  st.text("")
  st.subheader('Objective')
  st.markdown ('Analyze sales data of Super Store mart and identify opportunities to boost business growth.')

  
  st.text("")
  st.text("")
  st.markdown("<h1 style='font-size: 18px; color: black'><i/><b/>A bit of context...<i/><b/></h1>", unsafe_allow_html=True)
  st.write('Super Store is a small retail business located in the United States. They sell Furniture, Office Supplies and Technology products and their customers are the mass Consumer, Corporate and Home Offices. The data set contains sales, profit, geographical information of Super Store as well as other data attributes. The objective here is to analyze the sales data and identify weak areas and opportunities for Super Store to boost business growth.')
  
  st.text("")
  st.text("")
  st.subheader('Business Questions')
  st.write('_Some common business questions_')
  """
  *  What are the best-selling categories?
  *  What are the most profitable sub-categories?
  *  Which customer segment is the most profitable?
  *  Is Order Priority Affecting Our Sales Numbers?
  *  Which market has the highest number of sales?

  """


  st.text("")
  st.text("")
  text_col, image_col=st.columns([1, 0.35])
  text_col.subheader('**Interest Audience**')
  image_col.image('decision-making.png',width=230)

  text_col.write("""
  * CEO of the marketing agency
  * Owners of Superstore mart who are very interested in their business growth

  """)


if selected == "Data Exploration":
 
     st.header('Data Exploration') 

     st.text("")
     st.text("")

     st.sidebar.title('Side Panel')
     st.sidebar.write('Use this panel to upload, clean and explore your dataset.')
     st.sidebar.text("")


     #setup file upload
     file_upload = st.sidebar.file_uploader(label="Upload your CSV or Excel file (200MB max).", type=['csv', 'xlsx'])

     global df

     if file_upload is not None:
                print(file_upload)
                st.sidebar.markdown("<h1 style='font-size: 14px'>You successfully uploaded the datafile.</h1>", unsafe_allow_html=True)
                try:
                   df=pd.read_csv(file_upload)
                except Exception as e:
                   df=pd.read_excel(file_upload)

     if file_upload is None:
                st.sidebar.markdown("<h1 style='font-size: 14px'>Please upload your file to remove the error.</h1>", unsafe_allow_html=True)



     st.markdown('**Dataset Quick Look**')
                
     if st.checkbox ('Show Unprocessed Data'):
        st.write(df.head(10), use_column_width=None)
        st.write(df.shape)
        st.write('_This data is sourced from Kaggle._')
        st.text("")
        st.text("")
     if st.checkbox ('Statistical Description'):
        st.write(df.describe())
                



     st.markdown('**Multiple Options To Clean Your Dataset**')

      

     class MissingValues:

       def handle(self, df, _n_neighbors=3):
        # function for handling missing values in the data
        logger.info('Started handling of missing values...', self.missing_num.upper())
        start = timer()
        self.count_missing = df.isna().sum().sum()

        if self.missing_num: # numeric data
                logger.info('Started handling of NUMERICAL missing values... Method: "{}"', self.missing_num.upper())
        # mean, median or mode imputation
        elif self.missing_num in ['mean', 'median', 'most_frequent']:
                    imputer = SimpleImputer(strategy=self.missing_num)
                    df = MissingValues._impute_missing(self, df, imputer, type=['int64', 'int32'])
        # delete missing values
        elif self.missing_num == 'delete':
                    df = MissingValues._delete(self, df, type=['int32', 'int64'])
                    logger.debug('Deletion of {} NUMERIC missing value(s) succeeded', self.count_missing-df.isna().sum().sum())

        if self.missing_categ: # categorical data
                logger.info('Started handling of CATEGORICAL missing values... Method: "{}"', self.missing_categ.upper())

        # mode imputation
        elif self.missing_categ == 'most_frequent':
                    imputer = SimpleImputer(strategy=self.missing_categ)
                    df = MissingValues._impute(self, df, imputer, type='category')
        # delete missing values                    
        elif self.missing_categ == 'delete':
                    df = MissingValues._delete(self, df, type='category')
                    logger.debug('Deletion of {} CATEGORICAL missing value(s) succeeded', self.count_missing-df.isna().sum().sum())
        else:
              logger.debug('{} missing values found', self.count_missing)
        end = timer()
        logger.info('Completed handling of missing values in {} seconds', round(end-start, 6))
        return df

        df=handle(df)


     if st.checkbox ('Remove Missing Values (if any)'):
             st.subheader('Remove Missing Values')
             st.write(df, use_column_width=True)
             st.write(df.shape)
             st.write(df.isnull().sum())

     


     #extract year, month and day from datetime variables

     df['Order Date'] = pd.to_datetime(df['Order Date'],format='%Y%m%d')
     df['Order Year'] = pd.DatetimeIndex(df['Order Date']).year
     df['Order Month'] = pd.DatetimeIndex(df['Order Date']).month
     df['Order Day'] = pd.DatetimeIndex(df['Order Date']).day

     df['Ship Date'] = pd.to_datetime(df['Ship Date'],format='%Y%m%d')
     df['Ship Year'] = pd.DatetimeIndex(df['Ship Date']).year
     df['Ship Month'] = pd.DatetimeIndex(df['Ship Date']).month
     df['Ship Day'] = pd.DatetimeIndex(df['Ship Date']).day


     if st.checkbox ('Extract date from datetime variables'):
             st.subheader('Date Extraction')
             st.write(df, use_column_width=True) 
             st.write(df.shape)


     if st.checkbox('Remove Unnecessary Columns'):
        st.subheader('Remove Unnecessary Columns')
        df=df.drop(['Order ID','Row ID','Customer ID','State','Postal Code', 'Product ID', 'Discount','Shipping Cost', 'Region', 'Order Date', 'Ship Date', 'Ship Mode', 'City', 'Customer Name' ], axis = 1)
        st.write(df)
        st.write(df.shape)



     #def _compute_bounds(df, feature):
        # function that computes the lower and upper bounds for finding outliers in the data
        #cols_num = df.select_dtypes(include=np.number).columns
        
        #q1, q3 = np.percentile(cols_num, [25, 75])
        #iqr = q3 - q1

        #lb = q1 - (1.5 * iqr) 
        #ub = q3 + (1.5 * iqr) 


        #for feature in cols_num:
            #counter = 0 
            #for row_index,row_val in enumerate(df[feature]):
                #if row_val < lb or row_val > up:
                    #df = df.drop(row_val)
                    #counter +=1
            #df = df.reset_index(drop=True)
            #if counter != 0:
                #logger.debug('Deletion of {} outliers succeeded for feature "{}"', counter, feature)
        #return df

        #df=_compute_bounds(df)
                   

     #if st.sidebar.checkbox ('Remove Outliers'):
        #st.write(df, use_column_width=True)


    #traditional cleaning method for outliers removal

     for x in ['Sales']:
    # Calculate first and third quartile 
      q75,q25 = np.percentile(df.loc[:,x],[75,25])
   #Evaluate interquartile range 
      intr_qr = q75-q25
   #Estimate upper bound 
      max = q75+(1.5*intr_qr) 
   #Estimate lower bound 
      min = q25-(1.5*intr_qr) 
   #Replace data points that lie outside of the lower and upper bound with a null value 
      df.loc[df[x] < min,x] = np.nan 
      df.loc[df[x] > max,x] = np.nan

      df=df.dropna(subset=['Sales'])

     
     for x in ['Quantity']:
    # Calculate first and third quartile 
      q75,q25 = np.percentile(df.loc[:,x],[75,25])
   #Evaluate interquartile range 
      intr_qr = q75-q25
   #Estimate upper bound 
      max = q75+(1.5*intr_qr) 
   #Estimate lower bound 
      min = q25-(1.5*intr_qr) 
   #Replace data points that lie outside of the lower and upper bound with a null value 
      df.loc[df[x] < min,x] = np.nan 
      df.loc[df[x] > max,x] = np.nan
      df=df.dropna(subset=['Quantity'])   

     for x in ['Profit']:
    # Calculate first and third quartile 
      q75,q25 = np.percentile(df.loc[:,x],[75,25])
   #Evaluate interquartile range 
      intr_qr = q75-q25
   #Estimate upper bound 
      max = q75+(1.5*intr_qr) 
   #Estimate lower bound 
      min = q25-(1.5*intr_qr) 
   #Replace data points that lie outside of the lower and upper bound with a null value 
      df.loc[df[x] < min,x] = np.nan 
      df.loc[df[x] > max,x] = np.nan

      df=df.dropna(subset=['Profit']) 

      
     if st.checkbox('Remove Outliers'):
      st.subheader('Outliers Removal')
      st.write(df)
      st.write(df.shape)




if selected == 'Descriptive Analytics':


    # Remove whitespace from the top of the page and sidebar
     st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


     st.sidebar.title('Side Panel')
     st.sidebar.write('Use this panel to upload and visualize your dataset.')

    #setup file upload
     file_upload = st.sidebar.file_uploader(label="Upload your CSV or Excel file (200MB max).", type=['csv', 'xlsx'])


     if file_upload is not None:
                print(file_upload)
                st.sidebar.markdown("<h1 style='font-size: 14px'>You successfully uploaded the datafile.</h1>", unsafe_allow_html=True)
                try:
                   df_2=pd.read_csv(file_upload)
                except Exception as e:
                   df_2=pd.read_excel(file_upload)

     if file_upload is None:
                st.sidebar.markdown("<h1 style='font-size: 14px'>Please upload your file to remove the error.</h1>", unsafe_allow_html=True)

   
     if st.sidebar.checkbox('Processed Data'):
        st.write(df_2)
        st.subheader('Cleaned Data')
        st.write(df_2.shape)


      #dashboard title
     st.title('Global Superstore Sales & Profit Overall Performance')

      #top-level filters
     filter1,filter2,filter3=st.columns(3)
     df_2_sorted=df_2.sort_values(by='Order Year', ascending=True)
     df_3_sorted=df_2.sort_values(by='Order Month', ascending=True)
     df_4_sorted=df_2.sort_values(by='Order Day', ascending=True)
     year_filter = filter1.selectbox('Select desired year', pd.unique(df_2_sorted["Order Year"]))
     month_filter = filter2.selectbox('Select desired month', pd.unique(df_3_sorted["Order Month"]))
     day_filter = filter3.selectbox('Select desired day', pd.unique(df_4_sorted["Order Day"]))
    
     

      # creating a single-element container
     placeholder = st.empty()

      #dataframe filter
     df_2= df_2[df_2["Order Year"]== year_filter]
     df_2= df_2[df_2["Order Month"]== month_filter]
     df_2= df_2[df_2["Order Day"]== day_filter]
     
     

     st.text("")
     st.text("")

      

     #creating the kpis
     sum_sales= np.sum(df_2["Sales"])
     sum_profit= np.sum(df_2["Profit"])
     count_product_name= df_2['Product Name'].count()
     count_category= df_2['Category'].count()
     count_sub_category= df_2['Sub-Category'].count()
     profit_margin= (sum_profit/sum_sales)*100

    

     with placeholder.container():

        #create five columns
        kpi1,kpi2,kpi3,kpi4,kpi5,kpi6 = st.columns (6)

        #fill in the above five columns with respective metrics
        kpi1.metric(label="Total Sales ＄",
        value=int(sum_sales))

        kpi2.metric(label="Total Products",
        value= count_product_name)

        kpi3.metric(label="Total Profit ＄",
        value=int(sum_profit))

        kpi4.metric(label="Total Categories",
        value=count_category)

        kpi5.metric(label="Total Sub-Category",
        value=count_sub_category)

        kpi6.metric(label="Total Profit Margin %",
        value=int(profit_margin))



      # create two columns for charts
        fig_col1, fig_col2 = st.columns([1, 1.3])
        with fig_col1:
             st.markdown("<h1 style='text-align: center; font-size: 16px'>Which Market Has The Highest Number Of Sales?</h1>", unsafe_allow_html=True)
             fig = px.density_heatmap(
             data_frame=df_2, y="Sales", x='Market')
             fig.update_layout(showlegend=True, paper_bgcolor= background_color, width=450)
             fig_col1.plotly_chart(fig)
            
        with fig_col2:
             st.markdown("<h1 style='text-align:center;font-size: 16px'>What Is The Most Profitable Product Sub-Category?</h1>", unsafe_allow_html=True)
             fig2 = px.bar(data_frame=df_2, x="Sub-Category", y="Profit", orientation='v', color_discrete_sequence=['#250F91'])
             fig2.update_layout(showlegend=False, paper_bgcolor= background_color, width=800, autosize=True, xaxis_showgrid=False, yaxis_showgrid=False)
             fig_col2.plotly_chart(fig2)



    # create three columns for charts
     chart1,chart2,chart3 = st.columns([1,0.4,0.5])

        #display an altair chart
     with chart1:
       st.markdown("<h1 style='text-align: center;font-size: 16px'>How Are Our Sales Numbers Affected By Order Priority?</h1>", unsafe_allow_html=True)
       df= pd.DataFrame(df_2, columns=['Sales','Order Priority','Country'])
       x = alt.Chart(df_2).mark_circle().encode(x='Sales', y='Country', size='Order Priority',color='Order Priority', tooltip=['Sales', 'Order Priority', 'Country']) 
       chart1.altair_chart(x, use_container_width=True)

     
     with chart2:
         st.markdown("<h1 style='text-align:left;font-size: 16px'>What Are The Best-Selling Product Categories?</h1>", unsafe_allow_html=True)
         fig1 = px.pie(data_frame=df_2, values="Sales", names="Category",color='Category', color_discrete_map={'Technology':'#160473 ','Office Supplies':'#471D97 ', 'Furniture':'#FDF606 '})
         fig1.update_layout(showlegend=False,width=200, height=200, xaxis_showgrid=False, yaxis_showgrid=False,paper_bgcolor= background_color,font=dict(color='#383635', size= 15), margin=dict(l=1, r=1,b=1,t=1), autosize=True)
         chart2.plotly_chart(fig1)


     with chart3:
        st.markdown("<h1 style='text-align:left;font-size: 16px'>What Is The Most Profitable Customer Segment?</h1>", unsafe_allow_html=True)
        df_2=df_2.sort_values(by='Segment', ascending=False)
        fig2 = px.bar(data_frame=df_2, x="Segment", y="Profit",orientation='v', color_discrete_sequence=['#250F91'])
        fig2.update_layout(showlegend=False, paper_bgcolor= background_color,width=500, height=300, xaxis_showgrid=False, yaxis_showgrid=False)
        chart3.plotly_chart(fig2)





#Machine learning application

if selected == "Predictive Analytics":

   st.header('Predictive Look Into The Data')
   st.sidebar.title('Side Panel')
   st.sidebar.write('Use this panel to upload the cleaned dataset and dig deeper into its machine learning components.')
   st.sidebar.text("") 

   #setup file upload
   file_upload = st.sidebar.file_uploader(label="Upload your CSV or Excel file (200MB max).", type=['csv', 'xlsx'])
 
 
   if file_upload is not None:
                print(file_upload)
                st.sidebar.markdown("<h1 style='font-size: 14px'>You successfully uploaded the datafile.</h1>", unsafe_allow_html=True)
                try:
                   df_cleaned=pd.read_csv(file_upload)
                except Exception as e:
                   df_cleaned=pd.read_excel(file_upload)

   if file_upload is None:
                st.sidebar.markdown("<h1 style='font-size: 14px'>Please upload your file to remove the error.</h1>", unsafe_allow_html=True)
                



   #display encoded data

   train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=11, shuffle=True)

   #standard scale the data
   if st.sidebar.checkbox('Data Preprocessing'):
      st.text("")
      st.subheader('**Data Standardization**')
      scaler=StandardScaler()
      scaler=scaler.fit_transform(train_df[['Sales', 'Quantity', 'Profit', 'Order Year', 'Order Month', 'Order Day', 'Ship Year', 'Ship Month', 'Ship Day']])
      st.write(scaler)
      st.write('_Our training data is now transformed and scaled_.')
   

   encoder=ce.BinaryEncoder(cols=['Segment','Country', 'Market', 'Category', 'Sub-Category', 'Product Name', 'Order Priority'], return_df=True) 
   train_df_encoded=encoder.fit_transform(train_df)

   st.text("")
   st.text("")
   if st.sidebar.checkbox ('Correlation Matrix'):
     corr_matrix = train_df.corr()

     mask = np.triu(np.ones_like(corr_matrix,dtype = bool))
     fig=plt.figure(dpi=100)
     ax = plt.axes()
     sns.heatmap(corr_matrix,annot=True, mask=mask,lw=0,linecolor='white',fmt = "0.2f")
     st.subheader('Correlation Analysis')
     st.pyplot(fig)
     st.write('_Correlation coefficients between our numerical variables are now displayed. The stronger the correlation, the closer the correlation coefficient comes to 1._')


   st.sidebar.subheader ('Time to train our models!')


   if st.sidebar.checkbox ('Global Superstore Sales Encoded dataset'):
        st.subheader('**Data Encoding**')
        st.write(train_df_encoded)
        st.write('_The dataset was encoded for predictive analytics purposes._')

   
   st.sidebar.write('**Choose a model and tune their hyperparameters for an optimal model performance.**') 

    #defining hyperparameters for "Decision Tree Regressor" model

   if st.sidebar.checkbox('Decision Tree Regressor'):   

     st.markdown("<h1 style='font-size: 20px; color: black'><b/>Decision Tree Regressor Trained Model <b/></h1>", unsafe_allow_html=True)
 

     input_feature=st.text_input('Which feature should be used as the input feature?', 'Sales', key='input_feature')                 
    
     criterion=st.selectbox('What should be the optimal criterion?', options=['mse', 'poisson', 'friedman_mse'])

     splitter=st.selectbox ('What should be the optimal splitter?', options=['best', 'random'])

     max_depth = st.slider('What should be the optimal max_depth', min_value=2, max_value=8, step=2, key='max_depth')

     max_features=st.selectbox('What is the number of features to consider when looking for the best split?', options=['auto', 'sqrt', 'log2'], index=0, key='max_features')
   
     min_samples_split=st.slider('What is the minimum number of samples required to split an internal node?', min_value=2, max_value=8, step=2, key='min_samples_split')
   


     #train "Decision Tree Regressor" model    
     tree_regressor=DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, max_features=max_features, splitter=splitter)

     X_train_1=train_df_encoded.drop('Sales', axis=1)

     y_train_1=train_df_encoded[['Sales']]

     tree_regressor.fit(X_train_1,y_train_1)

     prediction=tree_regressor.predict(X_train_1)

    
     st.text("")

     results=mean_squared_error(y_train_1,prediction)

     results_rmse= np.sqrt(results)

     st.text("")
     st.text("")

     st.write('RMSE score of the model is:', results_rmse)

     st.text("")
     st.text("")
     st.text("")
     st.text("")

     st.markdown("<h1 style='text-align: center; font-size: 19px; color: #4682B4'><b/>We can now choose the best-performing model and proceed to testing phase!<b/></h1>", unsafe_allow_html=True)
     



     #defining hyperparameters for "Random Forest Regressor" model

   if st.sidebar.checkbox('Random Forest Regressor'):  

     st.markdown("<h1 style='font-size: 20px; color: black'><b/>Random Forest Regressor Trained Model <b/></h1>", unsafe_allow_html=True)
     

     input_feature=st.text_input('Which feature should be used as the input feature?', 'Sales', key='input_feature')
                   
     n_estimators=st.selectbox ('What is the optimal number of trees?', options=[100,200,300,400], index=0, key='n_estimators')

     criterion = st.selectbox('What is the quality of the split?', options=['mse', 'absolute_error', 'poisson'])

     max_depth = st.slider('What should be the optimal max_depth', min_value=2, max_value=8, step=2, key='max_depth')

     max_features=st.selectbox('What is the number of features to consider when looking for the best split?', options=['log2', 'sqrt', None], index=0, key='max_features')
   
     min_samples_split=st.slider('What is the minimum number of samples required to split an internal node?', min_value=2, max_value=8, step=2, key='min_samples_split')
   

     


     #train "Random Forest Regressor" model    
     random_forest_regressor=RandomForestRegressor(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, max_features=max_features, n_estimators=n_estimators)

     X_train_2=train_df_encoded.drop('Sales', axis=1)

     y_train_2=train_df_encoded[['Sales']]

     random_forest_regressor.fit(X_train_2,y_train_2)

     prediction=random_forest_regressor.predict(X_train_2)

    
     results=mean_squared_error(y_train_2,prediction)

     results_rmse= np.sqrt(results)

     st.text("")
     st.text("")

     st.write('RMSE score of the model is:', results_rmse)

     st.text("")
     st.text("")
     st.text("")
     st.text("")

     st.markdown("<h1 style='text-align: center; font-size: 19px; color: #4682B4'><b/>We can now choose the best-performing model and proceed to testing phase!<b/></h1>", unsafe_allow_html=True)

     


     #defining hyperparameters for "Gradient Boosting Regressor" model

   if st.sidebar.checkbox('Gradient Boosting Regressor'):

     st.markdown("<h1 style='font-size: 20px; color: black'><b/>Gradient Boosting Regressor Trained Model <b/></h1>", unsafe_allow_html=True)      

     input_feature=st.text_input('Which feature should be used as the input feature?', 'Sales', key='input_feature')               
    
     n_estimators=st.selectbox ('What is the optimal number of trees?', options=[100,200,300,400], index=0, key='n_estimators')

     criterion = st.selectbox('What is the quality of the split?', options=['mse', 'friedman_mse'])

     learning_rate=st.slider('What is the number of features to consider when looking for the best split?', min_value=0.01,max_value=0.2, step=0.04, key='learning_rate')
   
     min_samples_split=st.slider('What is the minimum number of samples required to split an internal node?', min_value=2, max_value=8, step=2, key='min_samples_split')

     max_depth = st.slider('What should be the optimal max_depth', min_value=2, max_value=8, step=2, key='max_depth')
   





     #train "Gradient Boosting Regressor" model    
     gradient_boosting_regressor=GradientBoostingRegressor(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, learning_rate=learning_rate, n_estimators=n_estimators)

     X_train_3=train_df_encoded.drop('Sales', axis=1)

     y_train_3=train_df_encoded[['Sales']]

     gradient_boosting_regressor.fit(X_train_3,y_train_3)

     prediction=gradient_boosting_regressor.predict(X_train_3)

     results=mean_squared_error(y_train_3,prediction)

     results_rmse= np.sqrt(results)

     st.text("")
     st.text("")

     st.write('RMSE score of the model is:', results_rmse)

     st.text("")
     st.text("")
     st.text("")
     st.text("")


     st.markdown("<h1 style='text-align: center; font-size: 19px; color: #4682B4'><b/>We can now choose the best-performing model and proceed to testing phase!<b/></h1>", unsafe_allow_html=True)
     



if selected == "Recommendations": 
    empty_col, pic_col=st.columns([5,10]) 
    pic_col.image('recommendations.png', width=300)
    st.text("")
    st.text("")
    st.markdown("<h1 style='text-align: center; color: #4682B4'><b/>Strategic Recommendations<b/></h1>", unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")


    """
    * Consumer and Corporate customer segments represent the highest proportion of the customerbase. They should look into targeting them, especially customers in the markets where we see highest amount of sales, by introducing special promotions and bundles for Mass Consumer and Home Offices, as well as sending emails or flyers.
    

    * For Home Offices, these customers might be busy with work and less likely to spend time selecting individual products, hence, creating a Home Office package with products used for offices such as tables, chairs, phone, copiers, storage, label, fasteners and bookcases.
     

    * They should focus on the Technology sub-category as well as Phones and Chairs as they are profitable and maybe bundle them with the less profitable products such as Chairs, Bookcases and more to offset the losses.
    

    * Based on the best-performing prediction model, this model will be tested on unseen data to check its accuracy and then predict future sales, which would also help the management in their data-driven decision making journey.

    """





     

     
 