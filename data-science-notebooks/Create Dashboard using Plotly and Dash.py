import dash
import more_itertools
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Load the data using pandas
data = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv"
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Dropdown options and year list
dropdown_options = [
    {"label": "Yearly Statistics", "value": "Yearly Statistics"},
    {"label": "Recession Period Statistics", "value": "Recession Period Statistics"},
]
year_list = [i for i in range(1980, 2024, 1)]

# Create the layout of the app
app.layout = html.Div([
    html.H1("Automobile Sales Statistics Dashboard",
            style={"color": "#503D36", "font-size": 24}),

    html.Div([
        html.Label("Select Statistics:"),
        dcc.Dropdown(
            id="dropdown-statistics",
            options=dropdown_options,
            value="Select Statistics",
            placeholder="Select a report type",
            style={
                "width": "80%",
                "padding": "3px",
                "font-size": "20px",
                "text-align-last": "center",
            },
        ),
    ]),

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id="select-year",
            options=[{"label": i, "value": i} for i in year_list],
            value="Select Year",
        ),
    ]),

    html.Div(id="output-container", className="chart-grid", style={"display": "flex"}),
])

# Define the callback to enable/disable year selection based on statistics type
@app.callback(
    Output(component_id="select-year", component_property="disabled"),
    Input(component_id="dropdown-statistics", component_property="value"),
)
def update_input_container(selected_statistic):
    if selected_statistic == "Yearly Statistics":
        return False
    else:
        return True

# Callback for plotting
@app.callback(
    Output(component_id="output-container", component_property="children"),
    [
        Input(component_id="select-year", component_property="value"),
        Input(component_id="dropdown-statistics", component_property="value"),
    ],
)
def update_output_container(selected_year, selected_statistic):
    if selected_statistic == "Recession Period Statistics":
        # Filter the data for recession periods
        recession_data = data[data["Recession"] == 1]

        yearly_rec = recession_data.groupby("Year")["Automobile_Sales"].mean().reset_index()
        R_chart1 = dcc.Graph(
            figure=px.line(
                yearly_rec,
                x="Year",
                y="Automobile_Sales",
                title="Average Automobile Sales fluctuation over Recession Period",
            )
        )

        average_sales = (
            recession_data.groupby("Vehicle_Type")["Automobile_Sales"]
            .mean()
            .reset_index()
        )
        R_chart2 = dcc.Graph(
            figure=px.bar(
                average_sales,
                x="Vehicle_Type",
                y="Automobile_Sales",
                title="Average Number of Vehicles Sold by Vehicle Type",
            )
        )

        exp_rec = (
            recession_data.groupby("Vehicle_Type")["Advertising_Expenditure"]
            .sum()
            .reset_index()
        )
        R_chart3 = dcc.Graph(
            figure=px.pie(
                exp_rec,
                names="Vehicle_Type",
                values="Advertising_Expenditure",
                title="Total Expenditure Share by Vehicle Type during Recessions",
            )
        )

        unemployment_rate = (
            recession_data.groupby("Vehicle_Type")["unemployment_rate"]
            .mean()
            .reset_index()
        )
        R_chart4 = dcc.Graph(
            figure=px.bar(
                unemployment_rate,
                x="Vehicle_Type",
                y="unemployment_rate",
                title="Effect of Unemployment Rate on Vehicle Type and Sales",
            )
        )

        return [
            html.Div(
                children=[R_chart1, R_chart2],
                style={"display": "flex"},
            ),
            html.Div(
                children=[R_chart3, R_chart4],
                style={"display": "flex"},
            ),
        ]

    elif selected_year and selected_statistic == "Yearly Statistics":
        yearly_data = data[data["Year"] == selected_year]

        yas = data.groupby("Year")["Automobile_Sales"].mean().reset_index()
        Y_chart1 = dcc.Graph(
            figure=px.line(
                yas, x="Year", y="Automobile_Sales", title="Yearly Automobile Sales"
            )
        )

        mas = data.groupby("Month")["Automobile_Sales"].mean().reset_index()
        Y_chart2 = dcc.Graph(
            figure=px.line(
                mas,
                x="Month",
                y="Automobile_Sales",
                title="Total Monthly Automobile Sales",
            )
        )

        avr_vdata = yearly_data.groupby("Vehicle_Type")["Automobile_Sales"].mean().reset_index()
        Y_chart3 = dcc.Graph(
            figure=px.bar(
                avr_vdata,
                x="Vehicle_Type",
                y="Automobile_Sales",
                title="Average Vehicles Sold by Vehicle Type in {}".format(selected_year),
            )
        )

        exp_data = (
            yearly_data.groupby("Vehicle_Type")["Advertising_Expenditure"]
            .mean()
            .reset_index()
        )
        Y_chart4 = dcc.Graph(
            figure=px.pie(
                exp_data,
                names="Vehicle_Type",
                values="Advertising_Expenditure",
                title="Total Advertisement Expenditure",
            )
        )

        return [
            html.Div(
                children=[Y_chart1, Y_chart2],
                style={"display": "flex"},
            ),
            html.Div(
                children=[Y_chart3, Y_chart4],
                style={"display": "flex"},
            ),
        ]
    else:
        return None

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
