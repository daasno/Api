import streamlit as st
import pandas as pd
import numpy as np
import math as mt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px

st.set_page_config(
    page_title="My Cig Quantities Prediction",
    initial_sidebar_state="expanded",
    menu_items={
         'Get Help': 'https://mycigmaroc.com/',
         'About': "This is app helps you estimate the Quantities to order"
    }
)


st.write(
    """
# My Cig Quantities Prediction HELPER
Upload the order Csv file for predictions.
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")
# ab_default = None
# result_default = None


def preprocessing(new_df):
    encoded_cat = ohe.fit_transform(new_df[['Catégorie']]).toarray()
    encoded_cat_labels = np.array(ohe.categories_).ravel()
    cat_ohe = pd.DataFrame(encoded_cat, columns=encoded_cat_labels)
    new_geek = pd.concat([new_df, cat_ohe], axis=1)
    new_geek.drop('Catégorie', inplace=True, axis=1)
    new_geek.drop('Marque', inplace=True, axis=1)
    new_geek.drop('Produit', inplace=True, axis=1)
    new_geek.drop('Fournisseur', inplace=True, axis=1)
    new_geek['Déclinaison'] = new_geek['Déclinaison'].fillna("None")
    encoded_dec = ohe.fit_transform(new_geek[['Déclinaison']]).toarray()
    encoded_dec_labels = np.array(ohe.categories_).ravel()
    dec_ohe = pd.DataFrame(encoded_dec, columns=encoded_dec_labels)
    new_geek = pd.concat([new_geek, dec_ohe], axis=1)
    new_geek.drop('Déclinaison', inplace=True, axis=1)
    new_geek.drop(['id', 'id decl'], inplace=True, axis=1)
    new_geek.drop(
        ['Ecart Casa Bourgogne', 'Ecart Rabat Kbibat', 'Ecart Casa Idriss-1er', 'Ecart Marrakech', 'Ecart Agadir',
         '30 j Stock central', 'Ecart Mycig Meknes', 'Ecart Fes', '60 j Stock central',
         'Ecart Rabat Hay Riad', 'Ecart Ecommerce', 'Ecart Ecommerce'], inplace=True, axis=1)
    new_geek['other_dec'] = new_geek['05'] + new_geek['0,7ohms'] + new_geek['0,25ohms'] + new_geek['Pods'] + new_geek[
        'Orange'] + new_geek['Rose'] + new_geek['Vert'] + new_geek['Rainbow'] + new_geek['1,4ohms'] + new_geek['Bleu'] + \
                            new_geek['Scarlet-Rouge'] + new_geek['0,4ohms'] + new_geek['Gold'] + new_geek['04'] + \
                            new_geek['Grey-Gris fonce'] + new_geek['Violet-Neon-Chestnut-Sky M'] + new_geek['0,6ohms'] + \
                            new_geek['065'] + new_geek['03'] + new_geek['Chrome'] + new_geek['None'] + new_geek['Noir']
    new_geek.drop(
        ['05', '0,7ohms', '0,25ohms', '1,4ohms', 'Pods', 'Orange', 'Rose', 'Vert', 'Rainbow', 'Bleu', 'Scarlet-Rouge',
         '0,4ohms', 'Gold', '04', 'Grey-Gris fonce', 'Violet-Neon-Chestnut-Sky M', '0,6ohms', '065', '03', 'Chrome',
         'None', 'Noir'], inplace=True, axis=1)
    new_geek['other_cat'] = new_geek['Cartouches vides pour pods'] + new_geek['Mods'] + new_geek['RBA'] + new_geek[
        'Drip tips'] + new_geek['Adaptateurs 510'] + new_geek['Pyrex'] + new_geek['Autres'] + new_geek['RTA'] + \
                            new_geek['Clearos'] + new_geek['Kits'] + new_geek['Fil Résistif'] + new_geek[
                                'RDA (Drippers)'] + new_geek['Mods electroniques']
    new_geek.drop(
        ['Cartouches vides pour pods', 'Mods', 'RBA', 'Drip tips', 'Adaptateurs 510', 'Pyrex', 'Autres', 'RTA',
         'Clearos', 'Kits', 'Fil Résistif', 'RDA (Drippers)', 'Mods electroniques'], inplace=True, axis=1)
    return new_geek


if uploaded_file:
    df = pd.read_csv("new_geek.csv")
    # The new_df got no quantities variable
    new_df = pd.read_csv(uploaded_file)
    names = new_df['Produit']
    decl = new_df['Déclinaison']
    cat = new_df['Catégorie']
    ##########Data Preprocessing##########
    reg = GradientBoostingRegressor(n_estimators=250, random_state=42,
                                    max_features='auto')
    ############Data preview##############
    st.markdown("### Data preview")
    st.dataframe(new_df.head(10))

    ohe = OneHotEncoder()
    new_geek = preprocessing(new_df)

    st.markdown("### Predict")
    #st.dataframe(new_geek.head(10))
    if st.button('Predict'):
        y = df.Quantities
        x = df.drop(columns=['Quantities'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        reg.fit(x_train, y_train)
        pred = reg.predict(new_geek)
        pred = np.ceil(pred)
        res = pd.concat([names, cat, decl, pd.DataFrame(pred, columns=['Predicted'])], axis=1)
        st.write(res)
        fig = px.pie(res, values='Predicted', names='Catégorie', title='Product categories')
        fig.show()
