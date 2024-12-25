# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:17:07 2021

@author: Andi5
"""
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from functions.analysis_and_plots import (preprocessing, train_test_split, plot_train_test_data,
seasonal_decompose_plot, exp_smoothing_model, exponential_smoothing_plot, rms_error_calc, load_model, predict_with_model, cumulative_addition, get_tables_from_sheet,create_df,sum_columns,subtract_dataframes)

st.set_page_config(page_title='FlowApp',  layout='wide', page_icon=':dollar:')

#this is the header
 

t1, t2 = st.columns((0.15,1)) 

t1.image('images/index.png', width = 120)
t2.title("Flow App - Cash Flow Forecasting")




## Data

with st.spinner('Updating Report...'):
    
    
   
    
    
    
    
    # Number of Completed Handovers by Hour
    
    g1, g2= st.columns((1,1))
    
    # Sample data
    data = {
        'Category': ['A', 'B', 'C', 'D'],
        'Values': [4500, 2500, 1050, 1500]
    }
    df = pd.DataFrame(data)

    # Title of the app
    g1.title("Interactive Pie Chart Example")

    # Pie chart using Plotly
    fig = px.pie(df, values='Values', names='Category', title='Distribution of Categories',
                color_discrete_sequence=px.colors.sequential.RdBu)

    # Update layout for better UX
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0, 0])  # Emphasize the first slice
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),  # Adjust margins
                    legend_title_text='Categories',  # Legend title
                    title_font=dict(size=20))  # Title font size

    # Display the chart
    g1.plotly_chart(fig, use_container_width=True) 

    
    # Predicted Number of Arrivals
    
    
    
    # Initialize session state to store transactions
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []

    # Function to add a new transaction
    def add_transaction(transaction):
        st.session_state.transactions.append(transaction)

    # Function to update a transaction
    def update_transaction(index, transaction):
        st.session_state.transactions[index] = transaction

    # Function to delete a transaction
    def delete_transaction(index):
        st.session_state.transactions.pop(index)

    # Function to render HTML table
    def render_html_table():
        transactions_html = """
        <h2>Simulation</h2>
        <table id="transactionTable">
            <thead>
                <tr>
                    <th>Transaction</th>
                    <th>Category</th>
                    <th>Amount</th>
                    <th>Date</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, transaction in enumerate(st.session_state.transactions):
            transactions_html += f"""
                <tr>
                    <td contenteditable="true" onblur="updateTransaction(this, {i})">{transaction[0]}</td>
                    <td>
                        <select onchange="updateCategory(this, {i})">
                            <option value="Loyer" {'selected' if transaction[1] == 'Loyer' else ''}>Loyer</option>
                            <option value="Restaurant" {'selected' if transaction[1] == 'Restaurant' else ''}>Restaurant</option>
                            <option value="Telephone" {'selected' if transaction[1] == 'Telephone' else ''}>Telephone</option>
                            <option value="Shopping" {'selected' if transaction[1] == 'Shopping' else ''}>Shopping</option>
                            <option value="Coffee" {'selected' if transaction[1] == 'Coffee' else ''}>Coffee</option>
                            <option value="Transports" {'selected' if transaction[1] == 'Transports' else ''}>Transports</option>
                            <option value="Electricte" {'selected' if transaction[1] == 'Electricte' else ''}>Electricte</option>
                            <option value="Netflix" {'selected' if transaction[1] == 'Netflix' else ''}>Netflix</option>
                            <option value="Divers Amazon" {'selected' if transaction[1] == 'Divers Amazon' else ''}>Divers Amazon</option>
                            <option value="Salle de sport" {'selected' if transaction[1] == 'Salle de sport' else ''}>Salle de sport</option>
                            <option value="Divers" {'selected' if transaction[1] == 'Divers' else ''}>Divers</option>
                            <option value="Autres" {'selected' if transaction[1] == 'Autres' else ''}>Autres</option>

                        </select>
                    </td>
                    <td contenteditable="true" onblur="updateTransaction(this, {i})">{transaction[2]}</td>
                    <td>
                        <input type="date" value="{transaction[3]}" onchange="updateDate(this, {i})">
                    </td>
                    <td><button class="btn btn-danger" onclick="deleteTransaction({i})">Delete</button></td>
                </tr>
            """
        
        transactions_html += """
            </tbody>
        </table>
        <button class="btn" onclick="addTransaction()">Add Row</button>

        <script>
            function addTransaction() {
                const newTransaction = ['', 'Food', 0, new Date().toISOString().split('T')[0]];
                const table = document.getElementById('transactionTable').getElementsByTagName('tbody')[0];
                const newRow = table.insertRow();
                newRow.innerHTML = `
                    <td contenteditable="true" onblur="updateTransaction(this, null)">${newTransaction[0]}</td>
                    <td>
                        <select onchange="updateCategory(this, null)">
                            <option value="Loyer">Loyer</option>
                            <option value="Restaurant">Restaurant</option>
                            <option value="Telephone">Telephone</option>
                            <option value="Shopping">Shopping</option>
                            <option value="Coffee">Coffee</option>
                            <option value="Transports">Transports</option>
                            <option value="Electricte">Electricte</option>
                            <option value="Netflix">Netflix</option>
                            <option value="Divers Amazon">Divers Amazon</option>
                            <option value="Salle de sport">Salle de sport</option>
                            <option value="Divers">Divers</option>
                            <option value="Autres">Autres</option>
                        </select>
                    </td>
                    <td contenteditable="true" onblur="updateTransaction(this, null)">${newTransaction[2]}</td>
                    <td>
                        <input type="date" onchange="updateDate(this, null)">
                    </td>
                    <td><button class="btn btn-danger" onclick="deleteTransaction(null)">Delete</button></td>
                `;
            }

            function updateTransaction(cell, index) {
                const row = cell.parentNode;
                const transaction = [
                    row.cells[0].innerText,
                    row.cells[1].querySelector('select').value,
                    parseFloat(row.cells[2].innerText) || 0,
                    row.cells[3].querySelector('input[type="date"]').value
                ];
                if (index !== null) {
                    const transactionData = JSON.stringify({ index: index, transaction: transaction });
                    // You would typically send this data to your backend here
                    console.log(transactionData);
                }
            }

            function updateCategory(select, index) {
                const row = select.parentNode.parentNode;
                const transaction = [
                    row.cells[0].innerText,
                    select.value,
                    parseFloat(row.cells[2].innerText) || 0,
                    row.cells[3].querySelector('input[type="date"]').value
                ];
                if (index !== null) {
                    const transactionData = JSON.stringify({ index: index, transaction: transaction });
                    // Send this data to your backend here
                    console.log(transactionData);
                }
            }

            function updateDate(input, index) {
                const row = input.parentNode.parentNode;
                const transaction = [
                    row.cells[0].innerText,
                    row.cells[1].querySelector('select').value,
                    parseFloat(row.cells[2].innerText) || 0,
                    input.value
                ];
                if (index !== null) {
                    const transactionData = JSON.stringify({ index: index, transaction: transaction });
                    // Send this data to your backend here
                    console.log(transactionData);
                }
            }

            function deleteTransaction(index) {
                const row = index === null ? event.target.parentNode.parentNode : document.querySelectorAll('#transactionTable tbody tr')[index];
                if (index !== null) {
                    // Send delete request to backend here
                    console.log(JSON.stringify({ index: index }));
                }
                row.parentNode.removeChild(row);
            }
        </script>
        <style>
            body,h2 {
                font-family: Arial, sans-serif;
                color: white;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #74e0e2;
            }
            .btn {
                padding: 10px 15px;
                border: none;
                background-color: #5638a3;
                color: white;
                cursor: pointer;
            }
            .btn-danger {
                background-color: #74e0e2;
            }
            .btn:hover {
                opacity: 0.9;
            }
            input[type="date"] {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
            }
        </style>
        """

        components.html(transactions_html, height=600)
    
    with g2:
        render_html_table()
    
    
      
    # Waiting Handovers table
    
    cw1, cw2 = st.columns((2.5, 1.7))
    
    
    cw1.write('')
    cw1.metric(label ='Actual Balance',value = 1000, delta = ' Compared to 1 hour ago', delta_color = 'inverse')
    cw1.write('')

    # Define the index
    index_depenses = [
        "Loyer", "Restaurant", "Telephone", "Shopping", "Coffee",
        "Transports", "Electricte", "Netflix", "Divers Amazon", "Salle de sport",
        "Divers", "Autres"
    ]

    index_revenus = ["Salaire"]

    # Load the Excel file
    file_path = 'CFF.xlsx'
    sheet_name = 'CFF'

    tables = get_tables_from_sheet(file_path, sheet_name)


    revenus = tables[0]
    depenses = tables[2]
    df_revenus = preprocessing(revenus)
    df_depense = preprocessing(depenses)

    dataframe_revenus = create_df(df_revenus,"REVENUS",index_revenus)
    dataframe_depenses = create_df(df_depense,"DEPENSES",index_depenses)
    total_depenses =  sum_columns(dataframe_depenses)
    total_revenus = sum_columns(dataframe_revenus)
    df_main = subtract_dataframes(total_revenus,total_depenses)
    train_data, test_data = train_test_split(df_main, split_index=24)
    preds_triple = exp_smoothing_model(train_data, test_data, model='triple')
    # Charger le modèle pré-entraîné
    loaded_model = load_model(filename='trained_exp_smoothing_model.pkl')



    # Prédire sur plusieurs semaines pour le nouveau client
    predictions = predict_with_model(loaded_model, df_main, weeks_ahead=4)

    flux_forecast = []

    flux_forecast = predictions['Predicted_Balance']


    forecast_balance = cumulative_addition(flux_forecast,1564.01)

        
    


    cw1.dataframe(dataframe_revenus)
    cw1.dataframe(dataframe_depenses)
    cw1.dataframe(df_main.transpose())
    cw1.dataframe(forecast_balance)
    
    with cw2 : 

        # Initialize a session state to store uploaded files
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        # File uploader
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
        if uploaded_file:
            # Save the uploaded file to the session state
            st.session_state.uploaded_files.append(uploaded_file)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Display the list of uploaded files
        st.subheader("Uploaded Files History")
        if st.session_state.uploaded_files:
            # Show the list of uploaded files
            
            for i, file in enumerate(st.session_state.uploaded_files, start=1):
                st.write(f"{i}. {file.name}")

           
            
        else:
            st.write("No files uploaded yet.")
# Contact Form

with st.expander("Contact us"):
    with st.form(key='contact', clear_on_submit=True):
        
        email = st.text_input('Contact Email')
        st.text_area("Query","Please fill in all the information or we may not be able to process your request")  
        
        submit_button = st.form_submit_button(label='Send Information')
    
    # LinkedIn URL
    linkedin_url = "https://www.linkedin.com/company/flowapp-eu/about/"
    # LinkedIn logo URL (you can also upload your own)
    linkedin_logo = "https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg"

    # Create a clickable LinkedIn button with logo
    st.markdown(
        f"[![LinkedIn]({linkedin_logo})]({linkedin_url})",
        unsafe_allow_html=True
    )
        
        
        
        
        
        
        
        
        
        