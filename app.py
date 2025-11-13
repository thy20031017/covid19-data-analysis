import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up the page configuration
st.set_page_config(
    page_title="CSV Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("📊 CSV Data Analyzer")
st.markdown("""
Upload your CSV file to view basic statistics and visualizations. 
This application supports automatic type detection, descriptive statistics, 
and interactive charts based on your data.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Main application logic
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    try:
        # Try to detect datetime columns automatically
        df = pd.read_csv(uploaded_file)
        
        # Display success message
        st.success("✅ File successfully uploaded!")
        
        # Show basic information about the dataset
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Number of rows: **{len(df)}**")
            st.write(f"Number of columns: **{len(df.columns)}**")
            
        with col2:
            st.subheader("Column Information")
            # Create a DataFrame with column info
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes,
                "Missing Values": df.isnull().sum()
            })
            st.dataframe(col_info)
        
        # Display sample of the data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        # Statistics section
        st.subheader("📈 Statistics")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to convert columns with date-like strings to datetime
        for col in df.columns:
            if col not in numerical_cols + datetime_cols:
                try:
                    # Test if the column can be converted to datetime
                    pd.to_datetime(df[col])
                    if st.checkbox(f"Treat '{col}' as datetime?"):
                        df[col] = pd.to_datetime(df[col])
                        datetime_cols.append(col)
                        if col in categorical_cols:
                            categorical_cols.remove(col)
                except:
                    pass
        
        # Display numerical statistics if there are any numerical columns
        if numerical_cols:
            st.write("### Numerical Columns Statistics")
            stats_expander = st.expander("Show Detailed Statistics")
            with stats_expander:
                st.dataframe(df[numerical_cols].describe())
            
            # Calculate and display additional statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Summary Statistics")
                total_rows = len(df)
                total_cells = df.size
                non_null_cells = df.count().sum()
                null_cells = total_cells - non_null_cells
                
                st.write(f"Total rows: **{total_rows:,}**")
                st.write(f"Total cells: **{total_cells:,}**")
                st.write(f"Non-null cells: **{non_null_cells:,}**")
                st.write(f"Null cells: **{null_cells:,}** ({null_cells/total_cells*100:.2f}%)")
                
                if numerical_cols:
                    # Calculate total sum for numerical columns if relevant
                    if st.checkbox("Show sum of numerical columns"):
                        sums = df[numerical_cols].sum()
                        st.dataframe(sums.to_frame("Total Sum"))
            
            with col2:
                # Correlation matrix for numerical columns
                if len(numerical_cols) > 1:
                    st.write("### Correlation Matrix")
                    corr = df[numerical_cols].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
                    st.pyplot(fig)
        
        # Display categorical column statistics
        if categorical_cols:
            st.write("### Categorical Columns")
            cat_col = st.selectbox("Select a categorical column to analyze", categorical_cols)
            if cat_col:
                st.write(f"**{cat_col} Distribution**")
                value_counts = df[cat_col].value_counts()
                
                # Display both as dataframe and bar chart
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(value_counts)
                with col2:
                    fig, ax = plt.subplots()
                    value_counts.plot(kind='bar', ax=ax)
                    plt.xticks(rotation=45)
                    plt.title(f"Distribution of {cat_col}")
                    st.pyplot(fig)
        
        # Display datetime column statistics if any
        if datetime_cols:
            st.write("### Datetime Columns")
            date_col = st.selectbox("Select a datetime column to analyze", datetime_cols)
            if date_col:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])
                
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                date_range = max_date - min_date
                
                st.write(f"**Date Range Analysis for {date_col}**")
                st.write(f"Start date: **{min_date.strftime('%Y-%m-%d')}**")
                st.write(f"End date: **{max_date.strftime('%Y-%m-%d')}**")
                st.write(f"Time span: **{date_range.days} days**")
        
        # Visualization section
        st.subheader("📊 Data Visualizations")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select a chart type",
            [
                "Time Series Plot",
                "Scatter Plot",
                "Histogram",
                "Bar Chart",
                "Line Chart",
                "Box Plot",
                "Pie Chart"
            ]
        )
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Time Series Plot
        if chart_type == "Time Series Plot":
            with col1:
                if datetime_cols:
                    date_col = st.selectbox("Select date column", datetime_cols)
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        df[date_col] = pd.to_datetime(df[date_col])
                else:
                    st.warning("No datetime columns found. Try converting a column to datetime in the statistics section.")
                    date_col = None
                
                if date_col and numerical_cols:
                    value_col = st.selectbox("Select value column", numerical_cols)
                    
                    # Optional grouping by categorical column
                    group_col = None
                    if categorical_cols:
                        if st.checkbox("Group by category"):
                            group_col = st.selectbox("Select category to group by", categorical_cols)
                    
                    # Optional aggregation
                    agg_func = st.selectbox(
                        "Aggregation function",
                        ["None", "Sum", "Mean", "Count", "Max", "Min"]
                    )
            
            if date_col and value_col and 'df' in locals():
                with col2:
                    st.write(f"**Time Series of {value_col} over {date_col}**")
                    
                    # Prepare data for plotting
                    plot_df = df.copy()
                    
                    # Apply aggregation if selected
                    if agg_func != "None":
                        if group_col:
                            # Group by date and category
                            if agg_func == "Sum":
                                plot_df = plot_df.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[value_col].sum().reset_index()
                            elif agg_func == "Mean":
                                plot_df = plot_df.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[value_col].mean().reset_index()
                            elif agg_func == "Count":
                                plot_df = plot_df.groupby([pd.Grouper(key=date_col, freq='D'), group_col]).size().reset_index(name=value_col)
                            elif agg_func == "Max":
                                plot_df = plot_df.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[value_col].max().reset_index()
                            elif agg_func == "Min":
                                plot_df = plot_df.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[value_col].min().reset_index()
                        else:
                            # Only group by date
                            if agg_func == "Sum":
                                plot_df = plot_df.groupby(pd.Grouper(key=date_col, freq='D'))[value_col].sum().reset_index()
                            elif agg_func == "Mean":
                                plot_df = plot_df.groupby(pd.Grouper(key=date_col, freq='D'))[value_col].mean().reset_index()
                            elif agg_func == "Count":
                                plot_df = plot_df.groupby(pd.Grouper(key=date_col, freq='D')).size().reset_index(name=value_col)
                            elif agg_func == "Max":
                                plot_df = plot_df.groupby(pd.Grouper(key=date_col, freq='D'))[value_col].max().reset_index()
                            elif agg_func == "Min":
                                plot_df = plot_df.groupby(pd.Grouper(key=date_col, freq='D'))[value_col].min().reset_index()
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if group_col:
                        # Plot multiple lines for each category
                        for category in plot_df[group_col].unique():
                            category_data = plot_df[plot_df[group_col] == category]
                            ax.plot(category_data[date_col], category_data[value_col], marker='o', linestyle='-', label=str(category))
                        ax.legend(title=group_col)
                    else:
                        # Plot a single line
                        ax.plot(plot_df[date_col], plot_df[value_col], marker='o', linestyle='-')
                    
                    plt.title(f'{value_col} vs {date_col}' + (f' by {group_col}' if group_col else ''))
                    plt.xlabel(date_col)
                    plt.ylabel(value_col)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
        
        # Scatter Plot
        elif chart_type == "Scatter Plot":
            if len(numerical_cols) >= 2:
                with col1:
                    x_col = st.selectbox("Select X-axis column", numerical_cols)
                    y_col = st.selectbox("Select Y-axis column", numerical_cols, index=1)
                    
                    # Optional color by categorical column
                    color_col = None
                    if categorical_cols:
                        if st.checkbox("Color by category"):
                            color_col = st.selectbox("Select category for color", categorical_cols)
                    
                    # Optional size by another numerical column
                    size_col = None
                    if len(numerical_cols) > 2:
                        if st.checkbox("Size by value"):
                            size_col = st.selectbox("Select column for bubble size", 
                                                 [col for col in numerical_cols if col not in [x_col, y_col]])
                
                with col2:
                    st.write(f"**Scatter Plot: {x_col} vs {y_col}**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if color_col:
                        # Create a scatter plot with color grouping
                        sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, size=size_col, ax=ax)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        # Simple scatter plot
                        if size_col:
                            sns.scatterplot(data=df, x=x_col, y=y_col, size=size_col, ax=ax)
                        else:
                            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                    
                    plt.title(f'{y_col} vs {x_col}')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            else:
                st.warning("Need at least two numerical columns for a scatter plot.")
        
        # Histogram
        elif chart_type == "Histogram":
            if numerical_cols:
                with col1:
                    hist_col = st.selectbox("Select column for histogram", numerical_cols)
                    bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
                    
                    # Optional grouping by categorical column
                    hist_group_col = None
                    if categorical_cols:
                        if st.checkbox("Group by category"):
                            hist_group_col = st.selectbox("Select category", categorical_cols)
                
                with col2:
                    st.write(f"**Histogram of {hist_col}**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if hist_group_col:
                        # Histogram with multiple groups
                        sns.histplot(data=df, x=hist_col, hue=hist_group_col, bins=bins, kde=True, ax=ax)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        # Simple histogram
                        sns.histplot(data=df, x=hist_col, bins=bins, kde=True, ax=ax)
                    
                    plt.title(f'Distribution of {hist_col}')
                    plt.xlabel(hist_col)
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            else:
                st.warning("No numerical columns found for histogram.")
        
        # Bar Chart
        elif chart_type == "Bar Chart":
            with col1:
                if categorical_cols:
                    bar_cat_col = st.selectbox("Select categorical column", categorical_cols)
                else:
                    st.warning("No categorical columns found.")
                    bar_cat_col = None
                
                if bar_cat_col and numerical_cols:
                    bar_val_col = st.selectbox("Select value column", numerical_cols)
                    
                    # Bar chart orientation
                    orientation = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True)
                    
                    # Optional aggregation for bar chart
                    bar_agg = st.selectbox(
                        "Aggregation",
                        ["Sum", "Mean", "Count", "Max", "Min"]
                    )
                else:
                    bar_val_col = None
            
            if bar_cat_col and bar_val_col:
                with col2:
                    st.write(f"**Bar Chart: {bar_val_col} by {bar_cat_col}**")
                    
                    # Prepare aggregated data
                    if bar_agg == "Count":
                        bar_data = df.groupby(bar_cat_col).size().reset_index(name=bar_val_col)
                    else:
                        if bar_agg == "Sum":
                            bar_data = df.groupby(bar_cat_col)[bar_val_col].sum().reset_index()
                        elif bar_agg == "Mean":
                            bar_data = df.groupby(bar_cat_col)[bar_val_col].mean().reset_index()
                        elif bar_agg == "Max":
                            bar_data = df.groupby(bar_cat_col)[bar_val_col].max().reset_index()
                        elif bar_agg == "Min":
                            bar_data = df.groupby(bar_cat_col)[bar_val_col].min().reset_index()
                    
                    # Sorting
                    bar_data = bar_data.sort_values(by=bar_val_col, ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if orientation == "Vertical":
                        sns.barplot(data=bar_data, x=bar_cat_col, y=bar_val_col, ax=ax)
                        plt.xticks(rotation=45, ha='right')
                    else:
                        sns.barplot(data=bar_data, y=bar_cat_col, x=bar_val_col, ax=ax)
                    
                    plt.title(f'{bar_agg} of {bar_val_col} by {bar_cat_col}')
                    plt.xlabel(bar_val_col if orientation == "Horizontal" else bar_cat_col)
                    plt.ylabel(bar_cat_col if orientation == "Horizontal" else bar_val_col)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
        
        # Line Chart
        elif chart_type == "Line Chart":
            if numerical_cols:
                with col1:
                    # Allow x-axis to be either numerical or datetime
                    available_x_cols = numerical_cols.copy()
                    if datetime_cols:
                        available_x_cols.extend(datetime_cols)
                    
                    line_x_col = st.selectbox("Select X-axis column", available_x_cols)
                    line_y_col = st.selectbox("Select Y-axis column", numerical_cols)
                    
                    # Optional grouping by categorical column
                    line_group_col = None
                    if categorical_cols:
                        if st.checkbox("Group by category"):
                            line_group_col = st.selectbox("Select category", categorical_cols)
                
                with col2:
                    st.write(f"**Line Chart: {line_y_col} vs {line_x_col}**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if line_group_col:
                        # Multiple lines for each category
                        for category in df[line_group_col].unique():
                            category_data = df[df[line_group_col] == category]
                            # Sort by x-axis to ensure line connects properly
                            category_data = category_data.sort_values(by=line_x_col)
                            ax.plot(category_data[line_x_col], category_data[line_y_col], 
                                    marker='o', linestyle='-', label=str(category))
                        ax.legend(title=line_group_col)
                    else:
                        # Single line plot
                        sorted_data = df.sort_values(by=line_x_col)
                        ax.plot(sorted_data[line_x_col], sorted_data[line_y_col], marker='o', linestyle='-')
                    
                    plt.title(f'{line_y_col} vs {line_x_col}')
                    plt.xlabel(line_x_col)
                    plt.ylabel(line_y_col)
                    
                    # Rotate x-axis labels if needed
                    if line_x_col in categorical_cols or (line_x_col in datetime_cols and df[line_x_col].nunique() > 12):
                        plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("No numerical columns found for line chart.")
        
        # Box Plot
        elif chart_type == "Box Plot":
            if numerical_cols:
                with col1:
                    box_val_col = st.selectbox("Select value column", numerical_cols)
                    
                    # Optional grouping by categorical column
                    box_cat_col = None
                    if categorical_cols:
                        if st.checkbox("Group by category"):
                            box_cat_col = st.selectbox("Select category", categorical_cols)
                
                with col2:
                    if box_cat_col:
                        st.write(f"**Box Plot: {box_val_col} by {box_cat_col}**")
                    else:
                        st.write(f"**Box Plot of {box_val_col}**")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if box_cat_col:
                        sns.boxplot(data=df, x=box_cat_col, y=box_val_col, ax=ax)
                        plt.xticks(rotation=45)
                    else:
                        sns.boxplot(data=df, y=box_val_col, ax=ax)
                    
                    plt.title(f'Box Plot of {box_val_col}' + (f' by {box_cat_col}' if box_cat_col else ''))
                    plt.xlabel(box_cat_col if box_cat_col else box_val_col)
                    plt.ylabel(box_val_col if box_cat_col else 'Value')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            else:
                st.warning("No numerical columns found for box plot.")
        
        # Pie Chart
        elif chart_type == "Pie Chart":
            if categorical_cols:
                with col1:
                    pie_col = st.selectbox("Select categorical column for pie chart", categorical_cols)
                    
                    # Optional value column for weighted pie chart
                    pie_val_col = None
                    if numerical_cols:
                        if st.checkbox("Weight by numerical column"):
                            pie_val_col = st.selectbox("Select value column", numerical_cols)
                
                with col2:
                    st.write(f"**Pie Chart of {pie_col}**")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    if pie_val_col:
                        # Create pie chart with weighted values
                        pie_data = df.groupby(pie_col)[pie_val_col].sum()
                    else:
                        # Create pie chart with counts
                        pie_data = df[pie_col].value_counts()
                    
                    # Limit to top categories if there are too many
                    if len(pie_data) > 8:
                        top_categories = pie_data.nlargest(7)
                        others = pd.Series([pie_data.sum() - top_categories.sum()], index=['Others'])
                        pie_data = pd.concat([top_categories, others])
                    
                    # Plot pie chart
                    pie_data.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
                    plt.ylabel('')  # Remove y-label
                    plt.title(f'Distribution of {pie_col}' + (f' by {pie_val_col}' if pie_val_col else ''))
                    
                    st.pyplot(fig)
            else:
                st.warning("No categorical columns found for pie chart.")
        
        # Add a note about customization
        st.markdown("""
        ### Customization Options
        - Select different columns for analysis
        - Choose aggregation methods (sum, mean, count, etc.)
        - Group data by categorical variables
        - Adjust chart parameters like bins for histograms
        """)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
else:
    # Display example dataset info
    st.info("📁 Upload a CSV file to begin analysis. For example, try the UM_C19_2021.csv dataset.")
    
    # Show a sample dataset preview if the COVID file exists
    import os
    if os.path.exists("UM_C19_2021.csv"):
        st.subheader("Sample Dataset Preview (UM_C19_2021.csv)")
        sample_df = pd.read_csv("UM_C19_2021.csv").head(5)
        st.dataframe(sample_df)