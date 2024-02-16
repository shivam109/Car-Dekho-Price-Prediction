import streamlit as st
import pandas as pd
import joblib


# Load the model
model = joblib.load(open('gradient_boosting_model.pkl', 'rb'))

# Demo dataset
demo_data = {
    'Transmission': [0],
    'Model_Year': [2015],
    'Seats': [5.0],
    'Kms_Driven': [120000.0],
    'Mileage(kmpl)': [23.1],
    'Max_Power(bhp)': [67.04],
    'Engine(CC)': [998.0],
    'Top_3_Features_Encoded': [931663.96007],
    'OEM_Encoded': [509677.447118],
    'Fifth Owner': [False],
    'First Owner': [False],
    'Fourth Owner': [False],
    'Second Owner': [False],
    'Third Owner': [True],
    'CNG': [False],
    'Diesel': [False],
    'Electric': [False],
    'LPG': [False],
    'Petrol': [True],
    'Bangalore': [True],
    'Chennai': [False],
    'Delhi': [False],
    'Hyderabad': [False],
    'Jaipur': [False],
    'Kolkata': [False],
}

demo_df = pd.DataFrame(demo_data)


def setting_bg():
    st.markdown(f"""<style>.stApp{{
                background:url("https://images.pexels.com/photos/7832995/pexels-photo-7832995.jpeg");
                background-size: cover;
                color: #333; /* Darker font color */
                font-weight: bold;}}
                
                </style>""", unsafe_allow_html=True)


setting_bg()


def main():
    st.title("Car Price Prediction App")

    # Get user input
    user_input = get_user_input()

    # Make predictions
    try:
        if st.button("Predict"):
            prediction = predict_price(user_input)
            if prediction > 0:
                st.balloons()
                styled_prediction = f"<div style='background-color:#8eff8e; padding:10px; border-radius: 10px;'><p style='color:#1e1e1e; font-size:20px; font-weight:bold;'>Predicted Price: {prediction:.2f} INR</p></div>"
                st.markdown(styled_prediction, unsafe_allow_html=True)

            else:
                st.warning(
                    '<p style="color:#ff6600; font-size:18px; font-weight:bold;">Sorry, No Cars available!</p>', unsafe_allow_html=True)
    except:
        st.warning('<p style="color:#ff0000; font-size:18px; font-weight:bold;">Something went wrong, Please try again!</p>', unsafe_allow_html=True)


def get_user_input():
    user_input = {}

    # Group columns
    city_columns = ['Bangalore', 'Chennai',
                    'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata']
    fuel_type_columns = ['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol']
    ownership_columns = ['First Owner', 'Second Owner',
                         'Third Owner', 'Fourth Owner', 'Fifth Owner']

    # Create dropdowns for grouped columns
    selected_city = st.selectbox("**Select City**", city_columns)
    selected_fuel_type = st.selectbox(
        "**Select Fuel Type**", fuel_type_columns)
    selected_ownership = st.selectbox(
        "**Select Ownership**", ownership_columns)
    encoded_feature_mapping = {
        931663.96006968: 'Power Steering, Power Windows Front, Air Conditioner',
        251114.28571429: 'Power Steering, Remote Trunk Opener, Air Conditioner',
        1771750.: 'Power Windows Front, Power Windows Rear, Leather Seats',
        278258.61538462: 'Power Steering, Remote Fuel Lid Opener, Air Conditioner',
        382451.80821918: 'Power Steering, Low Fuel Warning Light, Air Conditioner',
        58000.: 'Cup Holders Front, Navigation System, Tachometer',
        120000.: 'Power Steering, Power Windows Front',
        84851.5: 'Remote Trunk Opener, Remote Fuel Lid Opener, Tachometer',
        465000.: 'Accessory Power Outlet, Rear Seat Headrest, Heater',
        485000.: 'Accessory Power Outlet, Heater, Digital Odometer',
        822000.: 'Power Steering, Low Fuel Warning Light, Fabric Upholstery',
        515000.: 'Cup Holders Front, Tachometer, Glove Compartment',
        475000.: 'Accessory Power Outlet, Rear Seat Headrest, Air Conditioner',
        61000.: 'Remote Fuel Lid Opener, Low Fuel Warning Light, Heater',
        313500.: 'Power Windows Front, Power Windows Rear, Air Conditioner',
        211000.: 'Low Fuel Warning Light, Accessory Power Outlet, Air Conditioner',
        401000.: 'Low Fuel Warning Light, Accessory Power Outlet, Heater',
        516857.14285714: 'Power Steering, Power Windows Front, Digital Odometer',
        201666.66666667: 'Power Steering, Air Quality Control, Air Conditioner',
        1843750.: 'Low Fuel Warning Light, Tachometer, Fabric Upholstery',
        875000.: 'Power Windows Front, Power Windows Rear, Electronic Multi Tripmeter',
        75000.: 'Power Steering, Power Windows Front, Adjustable Head Lights',
        6475000.: 'Remote Fuel Lid Opener, Low Fuel Warning Light, Air Conditioner',
        183714.28571429: 'Power Steering, Vanity Mirror, Adjustable Steering',
        3749500.: 'Remote Trunk Opener, Remote Fuel Lid Opener, Air Conditioner',
        8765000.: 'Power Steering, Air Quality Control, Heater',
        89338.58333333: 'Power Steering, Multifunction Steering Wheel, Adjustable Steering',
        5700000.: 'Low Fuel Warning Light, Cup Holders Front, Air Conditioner',
        110000.: 'Drive Modes, Leather Seats, Digital Clock',
        105000.: 'Low Fuel Warning Light, Electronic Multi Tripmeter, Glove Compartment',
        124714.28571429: 'Low Fuel Warning Light, Vanity Mirror, Electronic Multi Tripmeter',
        550000: 'Low Fuel Warning Light, Rear Seat Headrest, Air Conditioner',
        6200000.: 'Remote Fuel Lid Opener, Low Fuel Warning Light, Fabric Upholstery',
        250000.: 'Remote Trunk Opener, Navigation System, Digital Odometer',
        94353.: 'Low Fuel Warning Light, Air Conditioner, Heater',
        276000.: 'Low Fuel Warning Light, Cup Holders Front, Digital Odometer',
        57000.: 'Remote Trunk Opener, Remote Fuel Lid Opener, Digital Odometer',
        700000.: 'Low Fuel Warning Light, Rear Seat Headrest, Tachometer',
        5895000.: 'Power Steering, Low Fuel Warning Light, Tachometer'}
    # selectbox for feature
    selected_features = st.selectbox("**Select Manufacturer**", list(
        encoded_feature_mapping.keys()), format_func=lambda x: encoded_feature_mapping[x])

    oem_mapping = {509677.44711757: 'Maruti', 739217.82178218: 'Ford', 769986.7575: 'Tata', 599452.30220147: 'Hyundai', 1874953.7037037: 'Jeep', 347810.81081081: 'Datsun', 639408.39263804: 'Honda', 877249.21135647: 'Mahindra', 3485806.62983425: 'BMW', 533941.36807818: 'Renault', 3604114.83253589: 'Mercedes-Benz', 2238564.62585034: 'Audi', 2859090.90909091: 'Mini', 1463900.76335878: 'Kia', 1039141.89189189: 'Skoda',
                   706169.55017301: 'Volkswagen', 3094405.40540541: 'Volvo', 1883307.69230769: 'MG', 1273750.0: 'Toyota', 577753.24675325: 'Nissan', 950000.0: 'Mahindra Ssangyong', 524727.27272727: 'Mitsubishi', 3277352.94117647: 'Jaguar', 300681.81818182: 'Fiat', 5437470.58823529: 'Land Rover', 258216.2027027: 'Chevrolet', 1951250.0: 'Citroen', 121600.0: 'Mahindra Renault', 1463800.0: 'Isuzu', 7300000.0: 'Lexus', 6590000.0: 'Porsche'}
    # selectbox for oem
    selected_oem = st.selectbox("**Select Manufacturer**", list(
        oem_mapping.keys()), format_func=lambda x: oem_mapping[x])

    # Transmission mapping
    transmission_mapping = {0: 'Manual', 1: 'Automatic'}
    # Selectbox for Transmission
    selected_transmission = st.selectbox("**Select Transmission**", list(
        transmission_mapping.keys()), format_func=lambda x: transmission_mapping[x])
    # Slider for Kms_Driven
    kms_driven_value = st.slider("**Kms Driven**", min_value=float(
        101), max_value=float(5500000), value=float(demo_df['Kms_Driven'].iloc[0]))
    # Slider for Mileage
    mileage_value = st.slider("**Mileage**", min_value=float(7.08), max_value=float(
        140), value=float(demo_df['Mileage(kmpl)'].iloc[0]))
    # Slider for Max_Power
    max_power_value = st.slider("**Max Power**", min_value=float(
        34.2), max_value=float(510), value=float(demo_df['Max_Power(bhp)'].iloc[0]))
    # Slider for Engine
    engine_value = st.slider("**Engine**", min_value=float(72), max_value=float(
        5000), value=float(demo_df['Engine(CC)'].iloc[0]))
    # Seats values
    available_seats = [5, 7, 4, 6, 8, 10, 9, 2]
    selected_seats = st.selectbox("**Select Seats**", available_seats)
    # Slider for Model_Year
    model_year_value = st.slider("**Model Year**", min_value=1985,
                                 max_value=2022, value=int(demo_df['Model_Year'].iloc[0]))

    # Iterate through columns and get user input
    for col in demo_df.columns:
        if col not in ['Price']:
            if col in city_columns:
                user_input[col] = (col == selected_city)
            elif col in fuel_type_columns:
                user_input[col] = (col == selected_fuel_type)
            elif col in ownership_columns:
                user_input[col] = (col == selected_ownership)
            elif col == 'Top_3_Features_Encoded':
                user_input[col] = selected_features
            elif col == 'OEM_Encoded':
                user_input[col] = selected_oem
            elif col == 'Transmission':
                user_input[col] = selected_transmission
            elif col == 'Seats':
                # Map selected Seats value
                user_input[col] = selected_seats
            elif col == 'Kms_Driven':
                user_input[col] = kms_driven_value
            elif col == 'Mileage(kmpl)':
                user_input[col] = mileage_value
            elif col == 'Max_Power(bhp)':
                user_input[col] = max_power_value
            elif col == 'Engine(CC)':
                user_input[col] = engine_value
            elif col == 'Model_Year':
                user_input[col] = model_year_value
            elif demo_df[col].dtype == 'bool':
                user_input[col] = st.checkbox(
                    f"Select {col}", value=bool(demo_df[col].iloc[0]))
            else:
                user_input[col] = st.number_input(
                    f"Enter {col}", value=float(demo_df[col].iloc[0]))

    return pd.DataFrame(user_input, index=[0])


def predict_price(user_input):
    prediction = model.predict(user_input)
    return prediction[0]


if __name__ == "__main__":
    main()
