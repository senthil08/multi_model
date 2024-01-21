import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

st.set_page_config(page_title="Multiple Prediction System", page_icon=":money_with_wings:")

# Create  the sidebar

with st.sidebar:
    select = option_menu(
        menu_title="Multiple Prediction System",
        options=["House Price Prediction", "Salary Prediction",
                 "Loan Prediction", "Car Price Prediction"],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )
if select == "House Price Prediction":
    # Set the title with the logo using HTML and CSS
    title_html = """
           <style>
               .title-container {
                   display: flex;
                   align-items: center;
                   justify-content: center;
               }
               .logo {
                   margin-right: 10px;
                   font-size: 48px; /* Adjust the font size of the logo here */
               }
           </style>
           <div class="title-container">
               <h1>House Price Prediction System</h1>
               <div class="logo">💸</div>
           </div>
       """
    st.markdown(title_html, unsafe_allow_html=True)

    house = pd.read_csv(r"dataset/House Price India.csv")

    x = house.drop(columns=["id", "Date", 'condition of the house', 'Area of the house(excluding basement)', 'Area of the basement',
                            'Postal Code', 'Lattitude', 'Longitude', 'living_area_renov', 'lot_area_renov', 'Price'], axis=1)
    y = house['Price']

    # 90% data - Train and 10% data - Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    from sklearn.linear_model import LinearRegression

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(x_train, y_train)
    training_data_prediction = lin_reg_model.predict(x_train)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        bed = st.slider("Number of Bedroom", 1, 10, 1)

        living_area = st.text_input("Living Area", placeholder="Ex - 34200 Square feet")

        bath = st.slider("Number of Bathroom", 1, 10, 1)

        lot_area = st.text_input("Lot Area", placeholder="Ex- 2500 Square feet")

    with col2:

        views = st.slider("Number of Views", 1, 5, 1)

        floor = st.text_input("Number of Floor", placeholder="Ex- 3 floor")

        grade = st.slider("Grade of the House", 1, 15, 1)

        water = st.selectbox("Waterfront Present", options=[1, 2])

    with col3:

        Distance = st.slider("Distance from the Airport", 1, 50, 2)

        build = st.text_input("Build Year", placeholder="2002")

        school = st.slider("Number of School nearby", 1, 50, 2)

        renovation = st.text_input("Renovation Year", placeholder="2015")

    st.write("")
    st.write("")
    col1, col2 = st.columns([0.438, 0.562])
    with col2:
        submit = st.button(label='Submit')

    if submit:
        try:
            userdata = np.array([[bed, bath, int(living_area), int(lot_area), int(floor), water, views, grade,
                                  int(build), int(renovation), school, Distance]])

            result = lin_reg_model.predict(userdata)
            st.success(f"The Price of the House is ( {result[0]:.2f} ).")
            st.balloons()

        except:
            st.warning('Please fill the all required information')


if select == "Salary Prediction":

    # Set the title with the logo using HTML and CSS
    title_html = """
        <style>
            .title-container {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .logo {
                margin-right: 10px;
                font-size: 48px; /* Adjust the font size of the logo here */
            }
        </style>
        <div class="title-container">
            <h1>Salary Prediction System</h1>
            <div class="logo">💸</div>
        </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

    # Rest of your Streamlit app code
    salary = pd.read_csv(r"dataset/Salary Data.csv")
    salary.dropna(inplace=True)
    salary.replace({"Gender": {"Male": 0, "Female": 1}}, inplace=True)
    salary.replace({"Education Level": {"Bachelor's": 0, "Master's": 1, "PhD": 2}}, inplace=True)
    salary.replace({"Domain": {"Data Analyst": 0, "Data Scientist": 1, "Python Devloper": 2, "Full stack developer": 3,
                               "Software Developer": 4, "UI UX Designer": 5, "ML Engineer": 6}}, inplace=True)

    x = salary.drop(columns=["Job Title", "Salary"], axis=1)
    y = salary["Salary"]

    # 90% data - Train and 10% data - Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    from sklearn.linear_model import LinearRegression

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(x_train, y_train)
    training_data_prediction = lin_reg_model.predict(x_train)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        age = st.text_input(label="Age")

        gender = st.selectbox(label="Gender", options=["male", "female"])
        gender_dict = {"male": 0.0, "female": 1.0}

    with col2:
        education = st.selectbox(label="Education", options=["UG", "PG", "phD"])
        education_dict = {"UG": 0.0, "PG": 1.0, "phD": 2.0}

        domain = st.selectbox(label="Job Title", options=['Data Analyst', 'Data Scientist', 'Python Developer',
                                                          'Full stack developer', 'Software Developer',
                                                          'UI UX Designer',
                                                          'ML Engineer'])
        domain_dict = {'Data Analyst': 0.0, 'Data Scientist': 1.0, 'Python Developer': 2.0, 'Full stack developer': 3.0,
                       'Software Developer': 4.0,
                       'UI UX Designer': 5.0, 'ML Engineer': 6.0}

    experience = st.slider("Years of Experience", 0, 50, 1)

    st.write("")
    st.write("")
    col1, col2 = st.columns([0.438, 0.562])
    with col2:
        submit = st.button(label='Submit')

    if submit:
        try:

            userdata = np.array(
                [[int(age), gender_dict[gender], education_dict[education], experience, domain_dict[domain]]])
            test_result = lin_reg_model.predict(userdata)
            st.success(f"The Salary of {domain} is  ({test_result[0]:.1f}) per month")
            st.balloons()
        except:
            st.warning('Please fill the all required information')

if select == "Loan Prediction":
    # Set the title with the logo using HTML and CSS
    title_html = """
        <style>
            .title-container {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .logo {
                margin-right: 10px;
                font-size: 48px; /* Adjust the font size of the logo here */
            }
        </style>
        <div class="title-container">
            <h1>Loan Prediction System</h1>
            <div class="logo">💸</div>
        </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)
    loan = pd.read_csv(r"dataset/train_u6lujuX_CVtuZ9i (1).csv")
    loan.dropna(inplace=True)
    loan.replace({"Gender": {"Male": 0, "Female": 1}}, inplace=True)
    loan.replace({"Married": {"Yes": 1, "No": 1}}, inplace=True)
    loan.replace({"Education": {"Graduate": 1, "Not Graduate": 0}}, inplace=True)
    loan.replace({"Self_Employed": {"Yes": 1, "No": 0}}, inplace=True)
    loan.replace({"Loan_Status": {"Y": 1, "N": 0}}, inplace=True)
    loan.replace({"Property_Area": {"Rural": 0, "Urban": 1, "Semiurban": 2}}, inplace=True)

    # feature and target selection

    x = loan.drop(
        columns=["Loan_ID", "CoapplicantIncome", "Dependents", "LoanAmount", "Loan_Amount_Term", "Loan_Status"], axis=1)
    y = loan["Loan_Status"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    from sklearn import svm

    svc_reg_model = svm.SVC()
    svc_reg_model.fit(x_train, y_train)
    training_data_prediction = svc_reg_model.predict(x_train)

    col1, col2 = st.columns(2, gap='large')
    with col1:
        gender = st.selectbox(label="Gender", options=["Male", "Female"])
        gender_dict = {"Male": 0.0, "Female": 1.0}

        married = st.selectbox(label="Married Status", options=["Yes", "No"])
        married_dict = {"Yes": 1.0, "No": 0.0}

        income = st.text_input(label="Income", placeholder="Enter the income in month:")

        education = st.selectbox(label="Education", options=["Graduate", "Not Graduate"])
        edu_dict = {"Graduate": 1.0, "Not Graduate": 0.0}

    with col2:
        self_employed = st.selectbox(label="Self Employed", options=["Yes", "No"])
        self_dict = {"Yes": 1.0, "No": 0.0}

        Credit_History = st.selectbox(label="Credict History", options=["Yes", "No"])
        credict_dict = {"Yes": 1.0, "No": 0.0}

        property_area = st.selectbox(label="Property Area", options=["Rural", "Urban", "Semiurban"])
        area_dict = {"Rural": 0, "Urban": 1, "Semiurban": 2}

    st.write("")
    st.write("")
    col1, col2 = st.columns([0.438, 0.562])
    with col2:
        submit = st.button(label='Submit')

    if submit:
        try:
            userdata = np.array([[gender_dict[gender], married_dict[married], edu_dict[education],
                                  self_dict[self_employed], int(income), credict_dict[Credit_History],
                                  area_dict[property_area]]])
            test_result_loan = svc_reg_model.predict(userdata)
            if test_result_loan[0] == 1:
                col1, col2, col3 = st.columns([0.33, 0.30, 0.35])
                with col2:
                    st.success('Loan Granted')
                st.balloons()

            else:
                col1, col2, col3 = st.columns([0.215, 0.57, 0.215])
                with col2:
                    st.error('Loan Denied')

        except:
            st.warning('Please fill the all required information')

if select == "Car Price Prediction":
    # Set the title with the logo using HTML and CSS
    title_html = """
        <style>
            .title-container {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .logo {
                margin-right: 10px;
                font-size: 48px; /* Adjust the font size of the logo here */
            }
        </style>
        <div class="title-container">
            <h1>Car Price Prediction System</h1>
            <div class="logo">💸</div>
        </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)
    car = pd.read_csv(r"dataset/car data.csv")
    le = LabelEncoder()
    car["Fuel_Type"] = le.fit_transform(car["Fuel_Type"])
    car["Transmission"] = le.fit_transform(car["Transmission"])
    car["Seller_Type"] = le.fit_transform(car["Seller_Type"])

    X = car.drop(["Car_Name", 'Selling_Price', "Present_Price", "Owner", "Kms_Driven"], axis=1)
    Y = car['Selling_Price']

    # 90% data - Train and 10% data - Test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

    from sklearn.linear_model import LinearRegression

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(x_train, y_train)
    training_data_prediction = lin_reg_model.predict(x_train)

    col1, col2 = st.columns(2, gap='large')
    with col1:
        Fuel = st.selectbox(label="Fuel Type", options=["Petrol", "CNG", "Diesel"])
        Fuel_dict = {"Petrol": 2.0, "Diesel": 1.0, "CNG": 0.0}

        year = st.text_input("Year")

    with col2:
        Transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
        Transmission_dict = {"Manual": 1.0, "Automatic": 0.0}

        seller = st.selectbox("Seller Type", options=["Dealer", "Individual"])
        Seller_dict = {"Dealer": 0.0, "Individual": 1.0}

    st.write("")
    st.write("")
    col1, col2 = st.columns([0.438, 0.562])
    with col2:
        submit = st.button(label='Submit')

    st.write("")

    if submit:
        try:

            Year = int(year)
            userdata = np.array([[Year, Fuel_dict[Fuel], Seller_dict[seller], Transmission_dict[Transmission]]])
            test_result = lin_reg_model.predict(userdata)
            st.success(f"The car price is {test_result[0]:.2f} Lak")
            st.balloons()
        except:
            st.warning('Please fill the all required information')
