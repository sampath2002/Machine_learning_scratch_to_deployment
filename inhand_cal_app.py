import streamlit as st 

from inhand_calculator import *

st.header("monthly inhand clculator")
st.title("employee inhand calculator")
st.caption("calculates inhand sal , monthly , annual , tax payable, rebate , pf ")
"""
    "ctc": int(input("enter the total CTC "))
    ,"tax": (input("enter the tas regime old/new")).lower()
    ,"country":(input("enter payroll country ")).lower()
    ,"bonus":(int(input("enter the bonus amount")))

"""
with st.form("tax_calculator"):
    data={
    "ctc": st.number_input("ctc", placeholder="enter the ctc"),
    "tax":st.selectbox("tax",["new","old"]),
    "country" : st.text_input("country", placeholder="enter the taxable country"),
    "bonus":st.number_input("bonus", placeholder="enter bonus amount")
    }

    submit = st.form_submit_button()
if submit:
    columns =["ctc","tax", "country","base","bonus"]
    table =inhand_generator(data, columns ).data_entry()
    st.dataframe(table)
    

