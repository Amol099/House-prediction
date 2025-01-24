import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
    df = generate_house_data(n_samples=100)
    x = df[['size']]
    y = df[['price']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LinearRegression()
    model.fit(x_train, y_train)

    return model

def main():
    st.title("Simple Linear Regression House Prediction App")
    st.write("Input your house size to estimate its price")

    model = train_model()

    size = st.number_input('House Size', min_value=500, max_value=2000, value=1500)

    if st.button('Predict Price'):
        predicted_price = model.predict([[size]])
        st.success(f'Estimated Price: ${predicted_price[0][0]:,.2f}')

        df = generate_house_data()

        fig = px.scatter(df, x="size", y="price", title="Size vs. House Price")
        fig.add_scatter(
            x=[size],
            y=[predicted_price[0][0]],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Prediction'
        )
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
