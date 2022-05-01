#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE




def process_ads_upload(): #Create File

    uploaded_file = st.file_uploader("Upload Files", type = ['csv'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name, "FileType":uploaded_file.type, "FileSize":uploaded_file.size}
        st.write(file_details)

        ADS_df = pd.read_csv(uploaded_file) #Read in Encoded ADS

        X = ADS_df.drop('Result', axis=1) #Drop result column and take all other columns in the table as the variable set
        y = ADS_df['Result'] #Use result as the target column

        sm = SMOTE(random_state=0)

        X_sm, y_sm = sm.fit_resample(X, y) #Run Smote

        X_sm["Result"]=y_sm #Add balanced result back into full dataset

        ADS_Rebalanced_df = X_sm

        st.dataframe(ADS_Rebalanced_df)

        st.download_button(label="Download File", data=ADS_Rebalanced_df.to_csv(), mime='text/csv', file_name='rebalanced_ads.csv')


def main():
    st.title("SMOTE Data Rebalancing")
    process_ads_upload()


if __name__ == "__main__":
    main()
