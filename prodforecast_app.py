import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
import time
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Constants
DAYS_PER_MONTH = 365.25 / 12

# Initialize session state
if 'stop' not in st.session_state:
    st.session_state.stop = False

def stop():
    st.session_state.stop = True

def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        required_columns = ['Well Series Name', 'Number Wells', 'Ultimate Recovery', 'Initial Rate', 
                            'Production Period', 'Individual well drilling Time', 'Rig Name', 'Start Drilling Time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def hyperbolic_decline(qi, di, b, t_days):
    t_months = t_days / DAYS_PER_MONTH
    return qi / (1 + b * di * t_months) ** (1 / b) if b != 0 else qi * np.exp(-di * t_months)

def objective(params, qi, eur, production_period_days):
    di, b = params
    days = np.arange(1, production_period_days + 1)
    production = hyperbolic_decline(qi, di, b, days)
    return abs(np.sum(production) - eur * 1e6)

def create_well_decline_parameters(initial_rate, ultimate_recovery, production_period_months):
    production_period_days = int(production_period_months * DAYS_PER_MONTH)
    result = minimize(objective, [0.1, 0.5], args=(initial_rate, ultimate_recovery, production_period_days), 
                      bounds=((1e-5, None), (1e-2, 0.99)))
    di, b = result.x
    days = np.arange(1, production_period_days + 1)
    daily_production = hyperbolic_decline(initial_rate, di, b, days)
    return di, b, daily_production

def get_monsoon_dates(start_date, end_date, year):
    start = datetime(year, start_date.month, start_date.day)
    end = datetime(year, end_date.month, end_date.day)
    if start > end:
        end = end.replace(year=year+1)
    return pd.date_range(start, end, freq='D')

def precompute_monsoon_months(monsoon_start, monsoon_end, reference_year, num_years=50):
    monsoon_start = datetime.strptime(monsoon_start, '%m/%d')
    monsoon_end = datetime.strptime(monsoon_end, '%m/%d')
    
    monsoon_dates = pd.DatetimeIndex([])
    for year in range(reference_year, reference_year + num_years):
        monsoon_dates = monsoon_dates.union(get_monsoon_dates(monsoon_start, monsoon_end, year))
    
    monsoon_months_df = pd.DataFrame({
        'Date': monsoon_dates,
        'Monsoon': 1
    })
    monsoon_months_df['Month'] = monsoon_months_df['Date'].dt.strftime('%Y-%m')
    
    return monsoon_months_df

def calculate_drilling_start_dates(df, reference_date, monsoon_start, monsoon_end, monsoon_strategy):
    reference_date = pd.to_datetime(reference_date)
    monsoon_months_df = precompute_monsoon_months(monsoon_start, monsoon_end, reference_date.year)
    monsoon_dates = set(monsoon_months_df['Date'].dt.strftime('%Y-%m-%d'))
    
    rig_current_date = {}
    start_dates = []

    for _, row in df.iterrows():
        rig_name = row['Rig Name']
        series_name = row['Well Series Name']
        number_of_wells = row['Number Wells']
        drilling_time_days = row['Individual well drilling Time'] * DAYS_PER_MONTH
        start_drilling_time_days = row['Start Drilling Time'] * DAYS_PER_MONTH
        
        if pd.isna(drilling_time_days):
            drilling_time_days = 0
        if pd.isna(start_drilling_time_days):
            start_drilling_time_days = 0
        
        if rig_name not in rig_current_date:
            rig_current_date[rig_name] = reference_date + timedelta(days=start_drilling_time_days)

        current_date = rig_current_date[rig_name]

        for well in range(1, number_of_wells + 1):
            well_name = f"{series_name}_well{well}"
            
            adjusted_drilling_time_days = drilling_time_days
            if monsoon_strategy == "Drill through (1.5x duration)":
                if current_date.strftime('%Y-%m-%d') in monsoon_dates:
                    adjusted_drilling_time_days *= 1.5
            elif monsoon_strategy == "Don't drill":
                while current_date.strftime('%Y-%m-%d') in monsoon_dates:
                    current_date += timedelta(days=1)

            start_dates.append({
                'Well Name': well_name,
                'Rig Name': rig_name,
                'Start Date': current_date,
                'Day Index': (current_date - reference_date).days
            })
            current_date += timedelta(days=adjusted_drilling_time_days)

        rig_current_date[rig_name] = current_date

    start_dates_df = pd.DataFrame(start_dates)
    return start_dates_df, monsoon_months_df

def generate_production_profile(df, start_dates_df):
    st.sidebar.write("Generating production profiles...")
    all_profiles = []
    params_list = []

    progress_bar = st.sidebar.progress(0)
    total_wells = len(start_dates_df)

    for index, (_, row) in enumerate(start_dates_df.iterrows()):
        if st.session_state.stop:
            break

        well_name = row['Well Name']
        rig_name = row['Rig Name']
        start_date = row['Start Date']
        day_index = row['Day Index']

        well_info = df[df['Well Series Name'].str.contains(well_name.split('_well')[0])]
        initial_rate = well_info['Initial Rate'].values[0]
        ultimate_recovery = well_info['Ultimate Recovery'].values[0]
        production_period_months = well_info['Production Period'].values[0]

        try:
            di, b, production_profile = create_well_decline_parameters(initial_rate, ultimate_recovery, production_period_months)
            params_list.append({'Well Name': well_name, 'Rig Name': rig_name, 'di': di, 'b': b})

            production_profile = [0] * day_index + list(production_profile)
            production_profile = production_profile[:int(production_period_months * DAYS_PER_MONTH)]
            production_profile = [round(p, 1) for p in production_profile]

            all_profiles.append(pd.DataFrame({well_name: production_profile}))

        except Exception as e:
            st.sidebar.write(f"Error processing {well_name}: {e}")

        progress_bar.progress((index + 1) / total_wells)

    progress_bar.empty()
    params_df = pd.DataFrame(params_list)

    st.sidebar.write("Aggregating production profiles...")
    production_profiles = pd.concat(all_profiles, axis=1)
    production_profiles['Total'] = production_profiles.sum(axis=1)
    production_profiles = production_profiles.loc[:production_profiles['Total'].ne(0)[::-1].idxmax()]
    production_profiles = production_profiles.drop(columns=['Total'])

    st.sidebar.write("Finished aggregating production profiles.")
    return params_df, production_profiles.fillna(0)

def convert_production_table_to_dates(production_table, reference_date):
    reference_date = pd.to_datetime(reference_date)
    daily_dates = [reference_date + timedelta(days=int(day)) for day in production_table.index]
    production_table_daily = production_table.copy()
    production_table_daily.index = daily_dates

    # Resample to monthly data, summing up the daily production
    production_table_monthly = production_table_daily.resample('M').sum()

    # Calculate the average daily rate for each month
    days_in_month = production_table_monthly.index.days_in_month
    production_table_monthly = production_table_monthly.divide(days_in_month, axis=0)

    return production_table_monthly

def plot_production(production_table_with_dates):
    series_columns = production_table_with_dates.columns.str.extract(r'^(.*)_well')[0].unique()
    series_data = pd.DataFrame()
    for series in series_columns:
        series_data[series] = production_table_with_dates.filter(like=series).sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    series_data.plot(kind='area', stacked=True, ax=ax, colormap='tab20')
    ax.set_xlabel('Date')
    ax.set_ylabel('Production (barrels/day)')
    ax.set_title('Stacked Production Chart by Well Series')
    
    handles, labels = ax.get_legend_handles_labels()
    ncol = min(6, len(labels))
    ax.legend(handles, labels, title='Well Series', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol)
    plt.subplots_adjust(bottom=0.2)
    st.pyplot(fig)
    
# Drilling sequence plotting 
def plot_drilling_sequence(start_dates_df, reference_date, df):
    # Convert reference_date to datetime if it's not already
    reference_date = pd.to_datetime(reference_date)
    
    # Sort the dataframe by Rig Name and Start Date
    start_dates_df = start_dates_df.sort_values(['Rig Name', 'Start Date'])
    
    # Get unique rig names and years
    rig_names = start_dates_df['Rig Name'].unique()
    years = sorted(start_dates_df['Start Date'].dt.year.unique())
    
    # Create a color map for well series
    well_series = df['Well Series Name'].unique()
    color_map = plt.cm.get_cmap('tab20')
    color_dict = {series: color_map(i/len(well_series)) for i, series in enumerate(well_series)}
    
    # Create the plot
    fig, axes = plt.subplots(len(years), 1, figsize=(20, 5*len(years)), squeeze=False)
    fig.suptitle('Drilling Sequence Visualization', fontsize=16)
    
    for year_idx, year in enumerate(years):
        ax = axes[year_idx, 0]
        ax.set_title(f'Year {year}')
        ax.set_xlim(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31))
        ax.set_ylim(-1, len(rig_names))
        ax.set_yticks(range(len(rig_names)))
        ax.set_yticklabels(rig_names)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        for month in range(1, 13):
            ax.axvline(pd.Timestamp(year=year, month=month, day=1), color='gray', linestyle='-', alpha=0.5)
        
        for idx, rig in enumerate(rig_names):
            rig_data = start_dates_df[(start_dates_df['Rig Name'] == rig) & (start_dates_df['Start Date'].dt.year == year)]
            for _, well in rig_data.iterrows():
                start = well['Start Date']
                end = start + pd.Timedelta(days=df[df['Well Series Name'].str.contains(well['Well Name'].split('_well')[0])]['Individual well drilling Time'].values[0] * DAYS_PER_MONTH)
                series = well['Well Name'].split('_well')[0]
                ax.barh(idx, (end - start).days, left=start, height=0.5, 
                        color=color_dict[series], alpha=0.8, 
                        edgecolor='black', linewidth=1)
                ax.text(start, idx, well['Well Name'], va='center', ha='left', fontsize=8, rotation=0)
        
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    plt.tight_layout()
    st.pyplot(fig)

    #Create a legend for well series
    legend_fig, legend_ax = plt.subplots(figsize=(8, 2))
    legend_ax.axis('off')
    for series, color in color_dict.items():
        legend_ax.bar(0, 0, color=color, label=series)
    legend_ax.legend(loc='center', ncol=3, title='Well Series')
    st.pyplot(legend_fig)







def main():
    st.title('Oil Production Forecast App')
    st.write('Load an Excel file with the required format:')
    st.write('Well Series Name, Number of Wells, Ultimate Recovery, Initial Rate, Production Period, Individual Well Drilling Time, Rig Name, Start Drilling Time (for each rig)')

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    reference_date = st.date_input("Reference Date", datetime.today())
    
    monsoon_start = st.text_input("Monsoon period - From (MONTH/DAY)", '10/01')
    monsoon_end = st.text_input("Monsoon period - To (MONTH/DAY)", '03/31')
    
    monsoon_strategy = st.selectbox(
        "Drilling Strategy During Monsoon:",
        ["Drill through (no penalty)", "Drill through (1.5x duration)", "Don't drill"]
    )

    compute_col, stop_col = st.columns([0.8, 0.2])
    with compute_col:
        compute_button = st.button('Compute')
    with stop_col:
        stop_button = st.button('Stop')

    if compute_button:
        if uploaded_file is not None:
            start_time = time.time()
            st.sidebar.write("Starting computation...")
            st.session_state.stop = False
            
            df = load_data(uploaded_file)
            if df is not None:
                start_dates_df, monsoon_months_df = calculate_drilling_start_dates(df, reference_date, monsoon_start, monsoon_end, monsoon_strategy)
                
                if not start_dates_df.empty:
                    st.write("### Monsoon Months")
                    st.write(monsoon_months_df)

                    st.write("### Drilling Start Dates")
                    st.write(start_dates_df)

                    params_df, production_table = generate_production_profile(df, start_dates_df)
                    production_table_monthly = convert_production_table_to_dates(production_table, reference_date)
                    end_time = time.time()

                    st.write("### Decline Parameters")
                    st.write(params_df)
                    
                    st.write("### Production Table (Daily)")
                    st.write(production_table)
                    
                    st.write("### Production Table with Dates (Monthly Average)")
                    st.write(production_table_monthly)

                    st.write("### Production Forecast Chart")
                    plot_production(production_table_monthly)
                    
                    st.write("### Drilling Sequence Visualization")
                    plot_drilling_sequence(start_dates_df, reference_date, df)
                    
                    st.sidebar.write(f"Total computation time: {end_time - start_time:.2f} seconds")
                else:
                    st.error("Error in calculating start dates. Please check the inputs and try again.")
      

    if stop_button:
        stop()

if __name__ == "__main__":
    main()