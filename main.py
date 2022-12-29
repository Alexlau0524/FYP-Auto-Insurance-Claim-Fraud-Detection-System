import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import time
import pickle
from streamlit_option_menu import option_menu


st.title("Auto Insurance Claim Fraud Detection Website")

df = pd.read_csv('Fraud_Insurance.csv')

df2 = df.loc[df['MonthClaimed']!='0']

df2.reset_index(drop=True, inplace=True)

df2["Age"] = df2["Age"].replace(0,df2["Age"].median())

df_Fraud = df2.loc[df2['FraudFound_P'] == 1]

X = df2[['Age']]

with open('RF_model.pkl', 'rb') as train_model: 
    model = pickle.load(train_model)

with open('RF_clr.dat', 'rb') as clr:
        RF_clr = pickle.load(clr)

with open('Model_comp.dat', 'rb') as mdp:
        Model_comp = pickle.load(mdp)

    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    choose = option_menu("Menu", ["Home", "Explore Dataset", "Visualize Dataset", "Model Performance", "Detect Fraud"],
                         styles={
        "menu-title":{"font-family": "Roboto", "font-size": "30px", "text-align": "left"},
        "container": {"padding": "4!important"},
        "nav-link": {"font-family": "Roboto","font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#bebebe"},
    }
    )

if choose == "Home":
    st.header("Home")
    st.write("---")
    st.subheader("What is Insurance Claim Fraud?")
    st.write("Insurance fraud, which is purposeful deceit conducted against an insurance company or agent for the goal of " + 
    "financial benefit, has received a lot of attention in recent years from governments, societies, and businesses. " + 
    "Especially in the field of auto insurance, auto insurance fraud cases account for about 80% of insurance fraud cases. " +  
    "It seriously infringes the rights of ordinary insurance consumers, disrupted the normal auto insurance market, " + 
    "and even endangers road safety.")

    st.write("---")
    col1, col2 = st.columns(2)
    with col1:     
        st.subheader("Why Insurance Claim Fraud Happens?")
        st.write("Claimants try to obtain benefits or insurance premiums from insurance companies that they do not have. "+
        "For instance, a person was in a car accident, although his vehicle was just slightly damaged. "+
        "However, the individuals made more money by lying about the severity of the accident in order to obtain greater insurance coverage. "+
        "Even some false insurance claims have occurred from accidents that haven’t happened yet.")
    with col2:
        st.subheader("Consequences of Insurance Claim Fraud")
        st.write("Insurance companies will suffer financial losses as a result of insurance claims fraud. "+
        "Insurance firms’ losses from auto insurance fraud are expected to exceed 14 billion dollar by 2020. "+
        "Insurance claim fraud hurts not just insurance companies’ financial interests, but also policyholders’ interests,"+
        " as well as other people’s property and the entire society. According to the CAIF (2003), "+
        "insurance fraud costs Americans at least 80 billion dollar per year, or roughly $950 per family. "+
        "As a result of the high costs of insurance fraud, many insurers pass the expenses on to consumers, raising insurance premiums. ")

if choose == "Explore Dataset":
    st.header("Explore Dataset")
    st.write("---")
    st.subheader("Insurance Claim Fraud Dataset")
    st.write("The dataset is collected from “Angoss Knowledge Seeker” software, which has a dataset of auto insurance records in the United States from 1994 to 1996. "+
    "The “FraudFound_P” target variable in this dataset is used to determine if a claim application is fraudulent or not. "+
    "This dataset has 15420 rows and 33 columns of data.")
    st.dataframe(df)

    st.write("---")
    st.subheader("Description of dataset")
    st.write("**Month** - Month of happening car accident")
    st.write("**WeekOfMonth** - Week of month for happening car accident in each Month")
    st.write("**DayOfWeek** - Day of week for happening car accident")
    st.write("**Make** - Car manufacturers ")
    st.write("**AccidentArea** - Car accident area")
    st.write("**DayOfWeekClaimed** - Day of week to claim for car insurance")
    st.write("**MonthClaimed** - Month to claim for car insurance")
    st.write("**WeekOfMonthClaimed** - Week of month to claim for car insurance")
    st.write("**Sex** - Gender of Claimant")
    st.write("**MaritalStatus** - Marital Status of Claimant")
    st.write("**Age** - Age of Claimant")
    st.text("---------------------------------------------------------------------------------------------------")
    st.write("**Fault** - Contain two types, Policy Holder and Third Party")
    st.markdown("Policy Holder - A policyholder (or policy holder) is the person who owns the insurance policy.")
    st.markdown("Third Party - A policy known as third-party insurance is one that the insured (first party) buys from the insurance provider"+
             " (second party) to defend themselves against claims made by third parties (third party)")
    st.text("---------------------------------------------------------------------------------------------------")
    st.write("**PolicyType** - Combination of VehicleCategory and BasePolicy")
    st.write("**VehicleCategory** - Category of Claimant's vehicle")
    st.write("**VehiclePrice** - Price of Claimant's vehicle")
    st.write("**FraudFound_P** - Target Variable. Detect if a claim application is fraudulent or not")
    st.write("**PolicyNumber** - Policy number is the unique number your insurance company uses to identify claimant account.")
    st.write("**Deductible** - The amount you pay for car insurance company before your insurance plan starts to pay")
    st.write("**DriverRating** - Driving history record")
    st.write("**Days_Policy_Accident** - The number of days from the inception of the policy to the occurrence of the accident")
    st.write("**Days_Policy_Claim** - The number of days from the inception of the policy to the claiming insurance")
    st.write("**PastNumberOfClaims** - Number of past insurance claims by claimant")
    st.write("**AgeOfVehicle** - Claimant's vehicle age")
    st.write("**AgeOfPolicyHolder** - Age Of Policy Holder (reference Fault)")
    st.write("**PoliceReportFiled** - The claimant has filed a car accident report with the police")
    st.write("**WitnessPresent** - Witnesses at the scene of the accident")
    st.write("**AgentType** - An agent is a person who represents an insurance firm and sells insurance policies on its behalf")
    st.write("**NumberOfSuppliments** - A supplement is required When the estimates don't factor in the cost of correcting further vehicle damage that wasn't initially identified.")
    st.write("**AddressChange_Claim** - Claimant changes his address")
    st.write("**NumberOfCars** - Number of Cars from Claimant")
    st.write("**Year** - Year for claim insurance")
    st.text("---------------------------------------------------------------------------------------------------")
    st.write("**BasePolicy** - ")
    st.markdown("1. Liability insurance: A type of insurance that offers defence against lawsuits brought by victims of injuries and property damage to others.")
    st.markdown("2. Collision insurance: A type of auto insurance that pays the insured for damage to their own car that was caused by the insured driver's negligence.")
    st.markdown("3. All Perils: All loss causes are covered by this optional coverage, with the exception of those expressly listed in your policy as exclusions.")
    st.text("---------------------------------------------------------------------------------------------------")
    st.write("---")
    st.subheader("Variable Data Type in dataset")
    st.write("There are 24 variables are object and 9 variables of integer in this dataset.")
    st.write(df.dtypes)

    st.write("---")
    st.subheader("Describe each variable (Without “Object” variable)")
    st.write("This “describe” function is used to summarise the auto insurance claim fraud dataset without “object” variable. "+
    "The criteria for summarise are count, mean, standard deviation (std), minimum, 25%, 50%, 75% and maximum. ")  
    st.write(df.describe().T)

    st.write("---")
    st.subheader("Target Variable")
    st.write("FraudFound_P” attribute is a class label. "+
    "The above graph indicates that “0” indicates no insurance claim fraud and “1” indicates insurance claim fraud.")
    st.write(df['FraudFound_P'].value_counts())

elif choose == "Visualize Dataset":
    st.header("Visualize Dataset (Dashboard)")
    st.write("---")
    select =  st.selectbox('Visualization the features',['Accident Area', 'Month Claimed', 'Gender', 'Marital Status', 'Age Number', 'Policy Type', 'Vehicle Price', 'Age of Vehicle'])
    if select == 'Age Number':
        st.subheader("Histogram for Age Number distribution")
        col1, col2 = st.columns([5,3])
        with col1:          
            fig = plt.figure(figsize=(7, 5))
            sns.histplot(data = df, x ='Age', stat='probability')
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph above that the distribution of “Age” attribute."+ 
            " Auto owners between 25 to 40 ages are the most active in auto insurance claims, comparing with others."+ 
            " Because the auto owners age between 25 to 40 are more likely to be involved in auto insurance claim, the auto insurance fraud will be more likely to happen on them. ")

        st.subheader("Histogram for relationship of Age and FraudFound_P")
        col3, col4 = st.columns([3,5])
        st.write("\n")
        with col3:
            st.write("Based on the graph above shows that the relationship between “Age” and “FraudFound_P” variables."+
            " Auto owners between the ages of 25 and 40 are most frequently involved in fraudulent insurance claims, comparing with others."+
            " Since there is a considerable likelihood that auto owners between the ages of 25 and 40 will be involved in fraudulent auto insurance claim activity,"+
            " the insurance company and insurance agent need to pay close attention to these drivers.")
        with col4:
            plt.figure(figsize=(7, 5))
            Age_Fraud = sns.histplot(x='Age', data=df_Fraud)
            Age_Fraud.set(xlabel="Age with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())
    
    elif select == 'Month Claimed':
        st.subheader("Barchart for Month Claimed distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(7, 5))
            sns.countplot(x='MonthClaimed',order=df2['MonthClaimed'].value_counts().index, data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “MonthClaimed” attribute."+ 
            " The month of the vehicle insurance claim is referred to as “MonthClaimed”."+ 
            " Since most people claim car insurance in January, May, and March, the likelihood of fraudulent auto insurance claims is highest in these three months.")
        
        st.write("\n")
        st.subheader("Histogram for relationship of MonthClaimed and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph above shows that the relationship between “MonthClaimed” and “FraudFound_P” variables."+
            " Fraudulent insurance claims are more likely to occur in May and March."+
            " In order to prevent auto insurance claim fraud, the insurance company and insurance agent must pay extra attention to monthly auto insurance claims in May and March. ")
        with col4:
            fig = plt.figure(figsize=(7, 5))
            sns.histplot(data = df_Fraud, x ='MonthClaimed', stat="probability", color = 'darkblue', kde=True)
            st.pyplot(fig)
    
    elif select == 'Accident Area':
        st.subheader("Barchart for Accident Area distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(7, 5))
            sns.countplot(x='AccidentArea', data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “AccidentArea” attribute."+
            " There are more car accidents in Urban areas than in Rural areas, which means that the likelihood of vehicle insurance fraud is higher in Urban areas. ")

        st.write("\n")
        st.subheader("Barchart for relationship of Accident Area and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph shows that the relationship between “AccidentArea” and “FraudFound_P” variables."+
            " Car accidents in Urban areas have resulted in more fraudulent insurance claims compare to in Rural areas. "+
            "Normally, the population of Urban areas is higher than Rural areas."+
            " Hence, there is a higher possibility for happening auto insurance claim fraud in Urban areas than Rural areas.")
        with col4:
            plt.figure(figsize=(7, 5))
            AccidentArea_Fraud = sns.barplot(x='AccidentArea', y = "FraudFound_P", data=df_Fraud, estimator=lambda x: len(x))
            AccidentArea_Fraud.bar_label(AccidentArea_Fraud.containers[0])
            AccidentArea_Fraud.set(xlabel="Accident Area with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())

    elif select == 'Policy Type':
        st.subheader("Barchart for PolicyType distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(15, 10))
            sns.countplot(x='PolicyType', data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “PolicyType” attributes."+ 
            " The “VehicleCategory” and “BasePolicy” variables are combined to provide the “PolicyType” variable."+ 
            " There are three different vehicle categories in this graph, with Sedan accounting for the majority of car insurance claims."+ 
            " Among sedan vehicles, Collision is the base policy type that is most likely to involve auto insurance claims.")

        st.write("\n")
        st.subheader("Barchart for relationship of PolicyType and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph shows that the relationship between “PolicyType” and “FraudFound_P” variables."+
            " Sedan is a vehicle category which is most likely to involve insurance claim fraud."+
            " Among sedan vehicles, All Perils is the base policy type that is most likely to involve auto insurance claims. ")
        with col4:
            plt.figure(figsize=(15, 10))
            PolicyType_Fraud = sns.barplot(x='PolicyType', y = "FraudFound_P", data=df_Fraud, estimator=lambda x: len(x))
            PolicyType_Fraud.bar_label(PolicyType_Fraud.containers[0])
            PolicyType_Fraud.set(xlabel="Policy Type with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())

    elif select == 'Gender':
        st.subheader("Barchart for Gender distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(7, 5))
            sns.countplot(x='Sex', data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “Sex” distribution."+ 
                        " Men are involved in car accidents more often than women."+ 
                        " Because men are more likely to be involved in car accidents, auto insurance fraud is more likely to happen to men.")

        st.write("\n")
        st.subheader("Barchart for relationship of Gender and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph shows that the relationship between “Sex” and “FraudFound_P” variables."+ 
            " Men are involved in insurance claim fraud more often than women."+ 
            " In order to avoid insurance claim fraud, the insurance company or insurance agent needs to pay extra attention to men.")
        with col4:
            plt.figure(figsize=(7, 5))
            Sex_Fraud = sns.barplot(x='Sex', y = "FraudFound_P", data=df_Fraud, estimator=lambda x: len(x))
            Sex_Fraud.bar_label(Sex_Fraud.containers[0])
            Sex_Fraud.set(xlabel="Gender with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())

    elif select == 'Marital Status':
        st.subheader("Barchart for Marital Status distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(7, 5))
            sns.countplot(x='MaritalStatus', data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “MaritalStatus” attribute."+ 
                    " The married person is the most often involve in car accidents, followed by single people and then divorcees and widows."+ 
                    " Because married people are more likely to be involved in car accidents, auto insurance fraud is more likely to happen on them.")

        st.write("\n")
        st.subheader("Barchart for relationship of Marital Status and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph shows that the relationship between “MaritalStatus” and “FraudFound_P” variables." +
            " The married person is the most often involve in insurance claim fraud, followed by single people and then divorcees and widows." +
            " Hence, insurance company or insurance agent require to pay attention on married person to avoid the insurance claim fraud happens. ")
        with col4:
            plt.figure(figsize=(7, 5))
            MaritalStatus_Fraud = sns.barplot(x='MaritalStatus', y = "FraudFound_P", data=df_Fraud, estimator=lambda x: len(x))
            MaritalStatus_Fraud.bar_label(MaritalStatus_Fraud.containers[0])
            MaritalStatus_Fraud.set(xlabel="Marital Status with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())
    
    elif select == 'Vehicle Price':
        st.subheader("Barchart for Vehicle Price distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(15, 10))
            sns.countplot(x='VehiclePrice', data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “VehiclePrice” attribute."+
            " Auto owners with auto prices between 20,000 and 29,000 are the most active in auto insurance claims, comparing with others."+
            " Because the auto owners with auto prices between 20,000 and 29,000 are more likely to be involved in auto insurance claim, the auto insurance fraud will be more likely to happen on them. ")

        st.write("\n")
        st.subheader("Barchart for relationship of Vehicle Price and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph above shows that the relationship between “VehiclePrice” and “FraudFound_P” variables."+
            " Auto owners with auto prices between 20,000 and 29,000 are the most involve in insurance claim fraud, comparing with others."+
            " Hence, the insurance company and insurance agent need to pay attention on auto owners with auto prices between 20,000 and 29,000 since they are high possibility involve in auto insurance claim fraud activity. ")
        with col4:
            plt.figure(figsize=(15, 10))
            VehiclePrice_Fraud = sns.barplot(x='VehiclePrice', y = "FraudFound_P", data=df_Fraud, estimator=lambda x: len(x))
            VehiclePrice_Fraud.bar_label(VehiclePrice_Fraud.containers[0])
            VehiclePrice_Fraud.set(xlabel="Vehicle Price with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())
    
    elif select == 'Age of Vehicle':
        st.subheader("Barchart for Age of Vehicle distribution")
        col1, col2 = st.columns([5,3])
        st.write("\n")
        with col1:          
            fig = plt.figure(figsize=(7, 5))
            sns.countplot(x='AgeOfVehicle', data=df)
            st.pyplot(fig)
        with col2:
            st.write("Based on the graph shows that the distribution of “AgeOfVehicle” attributes."+ 
            " The auto owners with autos 7 and over 7 years old are the most active on auto insurance claim, comparing with others."+ 
            " Because the auto owners with autos 7 and over 7 years old are more likely to be involved in auto insurance claim, the auto insurance fraud will be more likely to happen on them. ")

        st.write("\n")
        st.subheader("Barchart for relationship of Age of Vehicle and FraudFound_P")
        col3, col4 = st.columns([3,5]) 
        with col3:
            st.write("Based on the graph above shows that the relationship between “AgeOfVehicle” and “FraudFound_P” variables."+
            " Owners of vehicles with seven years old are the most involve in insurance claim fraud, comparing with others."+
            " Therefore, owners of vehicles with seven years need to be given special attention by the insurance company and insurance agent since they are more likely to be involved in fraudulent auto insurance claim activity.")
        with col4:
            plt.figure(figsize=(7, 5))
            AgeOfVehicle_Fraud = sns.barplot(x='AgeOfVehicle', y = "FraudFound_P", data=df_Fraud, estimator=lambda x: len(x))
            AgeOfVehicle_Fraud.bar_label(AgeOfVehicle_Fraud.containers[0])
            AgeOfVehicle_Fraud.set(xlabel="Age of Vehicle with Fraud Found", ylabel="Count")
            st.pyplot(plt.gcf())
    
elif choose == "Model Performance":
    st.header("Model Performance")
    st.write("---")
    st.subheader("Models Performance Comparison")
    st.dataframe(Model_comp)
    st.write("The Support Vector Machine, Random Forests, and K-Nearest Neighbor models compare in terms of evaluation metrics (Accuracy, Precision, Recall, and F1 score)."+
    " The best score for Accuracy, Precision, and F1-score obtained by Random Forests model. While the best score for Recall obtained by K-Nearest Neighbour model."+
    " This result showed that the Random Forests model had the best performance for the prediction of insurance claim fraud. Hence, the developer decided to select the Random Forests model for insurance claim fraud prediction. ")

    st.write("---")
    st.subheader("Best Model - Random Forest Classifier Model")
    st.write("A random forest model is utilised in the construction of this prediction system for insurance claim fraud.")

    st.write("There are 8 features has been choosen to build the model, such as “AccidentArea”, “MonthClaimed”, “Sex”, “MaritalStatus”, “Age”, “PolicyType”, “VehiclePrice”, and “AgeOfVehicle”."+
    " Those features are allowed people to clearly understand and comprehend them. ")
    st.write()
    st.markdown(f'<p style=font-size:18px;font-weight:bold;">Model Performance</p>', unsafe_allow_html=True)
    st.write("Model Accuracy: 95.4%")
    st.text("-------------------------------------------------------")
    st.markdown(f'<p style=font-size:16px;font-weight:bold;">Classification Report</p>', unsafe_allow_html=True)
    st.text(RF_clr)
    st.text("-------------------------------------------------------")
    st.write("The classification report above shows that the Random Forests Classifier model achieved 95% in precision,"+
    " 96% in recall, and 95% in F1-score in prediction of the fraudulent insurance claim. ")

elif choose == "Detect Fraud":
    st.header("Detect - Auto Insurance Claim Fraud")
    st.write('---')
    st.subheader('Specify Input Parameters')

    col1, col2 = st.columns(2)
    with col1:
        Sex = st.selectbox('Gender', ['Male', 'Female'])  
        Age = st.number_input('Age Number', value = 18, min_value=18, max_value=80)   
        AccidentArea = st.selectbox('Accident Area', ['Urban', 'Rural'])  
        AgeOfVehicle = st.selectbox('Age Of Vehicle', ['new', '2 years', '3 years', '4 years', '5 years', 
                                        '6 years', '7 years', 'more than 7'])               
    with col2:
        MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Widow', 'Divorced'])
        MonthClaimed = st.selectbox('Month Claimed - Claim Insurance Month',['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 
        PolicyType = st.selectbox('Policy Type', ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability',
                            'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision',
                            'Utility - Collision', 'Utility - Liability', 'Sport - All Perils']) 
        VehiclePrice = st.selectbox('Vehicle Price', ['less than 20000', '20000 to 29000', '20000 to 29000', 
                                             '30000 to 39000', '40000 to 59000', '60000 to 69000',
                                             'more than 69000']) 

    Newdf = pd.DataFrame(data={'AccidentArea': AccidentArea, 'MonthClaimed': MonthClaimed, 'Sex': Sex, 
                        'MaritalStatus': MaritalStatus, 'Age': Age, 'PolicyType': PolicyType, 
                        'VehiclePrice': VehiclePrice, 'AgeOfVehicle': AgeOfVehicle}, index = [0])

    Newdf['AccidentArea'] = Newdf['AccidentArea'].map({
                        'Urban': 1, 'Rural': 2})
    Newdf['MonthClaimed'] = Newdf['MonthClaimed'].map({
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12})
    Newdf['Sex'] = Newdf['Sex'].map({
                'Male': 1, 'Female': 2})
    Newdf['MaritalStatus'] = Newdf['MaritalStatus'].map({
                        'Single': 1, 'Married': 2, 'Widow': 3, 'Divorced': 4})
    Newdf['PolicyType'] = Newdf['PolicyType'].map({
                        'Sport - Liability': 1, 'Sport - Collision': 2, 'Sedan - Liability': 3,
                        'Utility - All Perils': 4, 'Sedan - All Perils': 5, 'Sedan - Collision': 6,
                        'Utility - Collision': 7, 'Utility - Liability': 8, 'Sport - All Perils': 9})
    Newdf['VehiclePrice'] = Newdf['VehiclePrice'].map({'less than 20000': 1, '20000 to 29000': 2, '20000 to 29000': 3, 
                                            '30000 to 39000': 4, '40000 to 59000': 5, '60000 to 69000': 6,
                                            'more than 69000': 7})
    Newdf['AgeOfVehicle'] = Newdf['AgeOfVehicle'].map({'new': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, 
                                                '6 years': 6, '7 years': 7, 'more than 7': 8})

    st.write("---")
    st.caption('Detection of Insurance claim fraud')
    if st.button("Detect"):
        result = model.predict(Newdf)
        with st.spinner('Wait for it...'):
            time.sleep(3) 
            if result == 0:
                st.markdown(f'<p style=color:#50C878;font-size:36px;font-weight:bold;">NO FRAUD</p>', unsafe_allow_html=True)
            elif result == 1:
                st.markdown(f'<p style=color:#FF0000;font-size:36px;font-weight:bold;">FRAUD !!!</p>', unsafe_allow_html=True)
            else:
                st.error("Please fill in all parameter")