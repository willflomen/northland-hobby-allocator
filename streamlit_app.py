import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import glob
import time
import base64
import zipfile
import io
import re
import matplotlib.pyplot as plt
import atexit
import shutil
import sys

# App version - helpful for troubleshooting
APP_VERSION = "1.1.0"

# Set page configuration
st.set_page_config(
    page_title="Camp Northland Hobby Allocator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the HobbyAllocator class directly instead of using subprocess
try:
    # Try to import the HobbyAllocator class from main_updated.py
    from main_updated import HobbyAllocator
    st.sidebar.success("✅ HobbyAllocator imported successfully")
except ImportError as e:
    st.sidebar.error(f"❌ Failed to import HobbyAllocator: {e}")
    st.error("""
    ### Error: HobbyAllocator import failed
    
    This application requires the HobbyAllocator class from main_updated.py.
    Please ensure main_updated.py is in the same directory as this app.
    """)
    st.stop()

# First, check for required dependencies
def check_dependencies():
    """Comprehensive check of all required dependencies"""
    missing_deps = []
    
    try:
        import pandas as pd
        st.sidebar.success("✅ pandas imported successfully")
    except ImportError:
        missing_deps.append("pandas")
        
    try:
        import numpy as np
        st.sidebar.success("✅ numpy imported successfully")
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import matplotlib.pyplot as plt
        st.sidebar.success("✅ matplotlib imported successfully")
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        import openpyxl
        st.sidebar.success("✅ openpyxl imported successfully")
    except ImportError:
        missing_deps.append("openpyxl")
        
    try:
        import pulp
        st.sidebar.success(f"✅ PuLP imported successfully (version {pulp.__version__})")
        
        # Test solver availability
        solvers = []
        
        # Test CBC
        try:
            cbc = pulp.PULP_CBC_CMD(msg=False)
            if cbc.available():
                solvers.append("CBC")
        except:
            pass
            
        # Test COIN
        try:
            coin = pulp.COIN_CMD(msg=False)
            if coin.available():
                solvers.append("COIN")
        except:
            pass
            
        # Test GLPK
        try:
            glpk = pulp.GLPK_CMD(msg=False)
            if glpk.available():
                solvers.append("GLPK")
        except:
            pass
        
        if solvers:
            st.sidebar.success(f"✅ Available solvers: {', '.join(solvers)}")
        else:
            st.sidebar.error("❌ No solvers available - optimization will fail")
            missing_deps.append("optimization solvers")
            
    except ImportError:
        missing_deps.append("pulp")
    
    if missing_deps:
        st.error(f"""
        ### Error: Missing dependencies
        
        The following required packages are missing: {', '.join(missing_deps)}
        
        If you're running this locally, please install them with:
        ```
        pip install {' '.join(missing_deps)}
        ```
        
        If you're seeing this on Streamlit Cloud, please contact the administrator.
        """)
        return False
    
    return True

# Helper functions - defined at the top so they're available throughout the file
def choice_to_word(choice):
    words = {
        1: "First",
        2: "Second",
        3: "Third",
        4: "Fourth",
        5: "Fifth"
    }
    return words.get(choice, str(choice) + "th")

# Function to create a zip file containing all files
def create_zip_of_files(files_dict):
    """Create a zip file in memory containing all the specified files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_dict.items():
            if isinstance(content, pd.DataFrame):
                # Save DataFrame to CSV in memory
                csv_buffer = io.StringIO()
                content.to_csv(csv_buffer, index=False)
                zip_file.writestr(filename, csv_buffer.getvalue())
            elif isinstance(content, bytes):
                # Save binary content (like images)
                zip_file.writestr(filename, content)
            else:
                # Save string content
                zip_file.writestr(filename, str(content))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Create plotting functions
def create_allocation_bar_chart(summary_df):
    """Create an allocation bar chart"""
    # Extract data for plotting
    hobbies = summary_df['Hobby Name'].tolist()
    assigned = summary_df['Number of Campers Assigned'].tolist()
    min_capacities = summary_df['Min Capacity'].tolist()
    max_capacities = summary_df['Max Capacity'].tolist()
    
    # Check if Pre-assigned Campers column exists
    has_preassigned = False
    regular_assigned = assigned.copy()
    pre_assigned = [0] * len(assigned)
    
    if 'Pre-assigned Campers' in summary_df.columns:
        has_preassigned = True
        pre_assigned = summary_df['Pre-assigned Campers'].tolist()
        if 'Regular Assigned Campers' in summary_df.columns:
            regular_assigned = summary_df['Regular Assigned Campers'].tolist()
        else:
            # Calculate regular assigned if not present
            regular_assigned = [a - p for a, p in zip(assigned, pre_assigned)]
    
    # Check if Restricted column exists
    has_restricted = False
    restricted_status = []
    
    if 'Restricted' in summary_df.columns:
        has_restricted = True
        restricted_status = summary_df['Restricted'].tolist()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the stacked bars
    bar_width = 0.6
    bar_positions = range(len(hobbies))
    
    # Create stacked bars for pre-assigned and regular campers
    if has_preassigned:
        bars1 = ax.bar(bar_positions, regular_assigned, bar_width, label='Regular Assigned Campers', color='#3498db')
        bars2 = ax.bar(bar_positions, pre_assigned, bar_width, bottom=regular_assigned, label='Pre-assigned Campers', color='#9b59b6')
    else:
        # Just one bar if no pre-assignment data
        bars1 = ax.bar(bar_positions, assigned, bar_width, label='Assigned Campers', color='#3498db')
    
    # Add count labels on top of bars
    for i, count in enumerate(assigned):
        ax.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Plot min and max capacity lines
    min_cap_line = None
    max_cap_line = None
    for i, (min_cap, max_cap) in enumerate(zip(min_capacities, max_capacities)):
        min_line = ax.plot([i - bar_width/2, i + bar_width/2], [min_cap, min_cap], 'r-', linewidth=2)
        max_line = ax.plot([i - bar_width/2, i + bar_width/2], [max_cap, max_cap], 'g-', linewidth=2)
        
        if i == 0:
            min_cap_line = min_line[0]
            max_cap_line = max_line[0]
    
    # Mark restricted hobbies if applicable
    if has_restricted:
        for i, status in enumerate(restricted_status):
            if status == "Yes":
                ax.text(i, -2, "*", ha='center', va='top', fontsize=24, color='red')
        
        if any(status == "Yes" for status in restricted_status):
            ax.text(0.01, 0.01, "* Restricted hobbies (only pre-assigned campers)", transform=ax.transAxes, 
                   fontsize=10, color='red', ha='left')
    
    # Set axis labels and title
    ax.set_xlabel('Hobby', fontsize=12)
    ax.set_ylabel('Number of Campers', fontsize=12)
    ax.set_title('Camper Allocation by Hobby', fontsize=16)
    
    # Set x-axis tick labels to hobby names
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(hobbies, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    if has_preassigned:
        ax.legend([bars1, bars2, min_cap_line, max_cap_line], 
                ['Regular Assigned Campers', 'Pre-assigned Campers', 'Min Capacity', 'Max Capacity'])
    else:
        ax.legend([bars1, min_cap_line, max_cap_line], 
                ['Assigned Campers', 'Min Capacity', 'Max Capacity'])
    
    # Add grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_choice_distribution_chart(choice_data):
    """Create a choice distribution chart"""
    # Process choice distribution data
    choices = ['First Choice', 'Second Choice', 'Third Choice', 'Fourth Choice', 'Fifth Choice', 'No Choice Match']
    counts = [
        choice_data.get(1, 0),  # First choice
        choice_data.get(2, 0),  # Second choice
        choice_data.get(3, 0),  # Third choice
        choice_data.get(4, 0),  # Fourth choice
        choice_data.get(5, 0),  # Fifth choice
        choice_data.get(0, 0),  # No choice match
    ]
    
    # Add pre-assigned if available
    if 'pre' in choice_data and choice_data['pre'] > 0:
        choices.append('Pre-assigned')
        counts.append(choice_data['pre'])
    
    # Calculate percentages based on the total
    total_campers = sum(counts)
    percentages = [count / total_campers * 100 for count in counts] if total_campers > 0 else [0] * len(counts)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for the bars
    colors = ['#2ecc71', '#27ae60', '#3498db', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
    
    # Plot the bars
    bar_positions = range(len(choices))
    bars = ax.bar(bar_positions, counts, color=colors[:len(choices)])
    
    # Add count and percentage labels on top of bars
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        if count > 0:  # Only add label if there are campers in this category
            ax.text(i, count + 2, f"{count}\n({percentage:.1f}%)", ha='center', va='bottom', fontweight='bold')
    
    # Set axis labels and title
    ax.set_xlabel('Choice Rank', fontsize=12)
    ax.set_ylabel('Number of Campers', fontsize=12)
    ax.set_title('Distribution of Choice Ranks Assigned', fontsize=16)
    
    # Set x-axis tick labels
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(choices, rotation=0, fontsize=10)
    
    # Add grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_hobby_choice_distribution_chart(hobby_choice_counts, restricted_hobbies=[]):
    """Create a stacked bar chart showing the distribution of choice ranks for each hobby"""
    
    # Get hobbies and their data
    hobbies = list(hobby_choice_counts.keys())
    
    # Extract data for plotting
    data = {
        '1st Choice': [hobby_choice_counts[h].get(1, 0) for h in hobbies],
        '2nd Choice': [hobby_choice_counts[h].get(2, 0) for h in hobbies],
        '3rd Choice': [hobby_choice_counts[h].get(3, 0) for h in hobbies],
        '4th Choice': [hobby_choice_counts[h].get(4, 0) for h in hobbies],
        '5th Choice': [hobby_choice_counts[h].get(5, 0) for h in hobbies],
        'No Choice': [hobby_choice_counts[h].get(0, 0) for h in hobbies],
        'Pre-assigned': [hobby_choice_counts[h].get('pre', 0) for h in hobbies]
    }
    
    # Get total counts for each hobby for sorting
    total_counts = [sum(hobby_choice_counts[h].values()) for h in hobbies]
    
    # Sort hobbies by total campers (descending)
    sorted_indices = np.argsort(total_counts)[::-1]
    hobbies = [hobbies[i] for i in sorted_indices]
    total_counts = [total_counts[i] for i in sorted_indices]
    
    # Reorder the data after sorting
    for key in data:
        data[key] = [data[key][i] for i in sorted_indices]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define colors for the stacked bars
    colors = {
        '1st Choice': '#2ecc71',  
        '2nd Choice': '#27ae60',  
        '3rd Choice': '#3498db', 
        '4th Choice': '#f1c40f', 
        '5th Choice': '#e67e22', 
        'No Choice': '#e74c3c', 
        'Pre-assigned': '#9b59b6'
    }
    
    # Create the stacked bars
    bottom = np.zeros(len(hobbies))
    bars = {}
    
    # Draw bars in specific order (1st choice at bottom, pre-assigned at top)
    stacking_order = ['1st Choice', '2nd Choice', '3rd Choice', '4th Choice', '5th Choice', 'No Choice', 'Pre-assigned']
    
    for category in stacking_order:
        bars[category] = ax.bar(hobbies, data[category], bottom=bottom, label=category, color=colors[category])
        bottom += np.array(data[category])
    
    # Add total count labels on top of stacked bars
    for i, total in enumerate(total_counts):
        if total > 0:  # Only label bars with campers
            ax.text(i, total + 1, str(total), ha='center', va='bottom', fontweight='bold')
    
    # Add data values to each segment of the stacked bars
    bottom_tracker = np.zeros(len(hobbies))
    for category in stacking_order:
        for i, value in enumerate(data[category]):
            if value > 0:  # Only label segments with campers
                # Calculate the middle position of the segment for text placement
                pos = bottom_tracker[i] + value/2
                # Add text with smaller font if the segment is narrow
                if value > 2:  # Only add text if segment is tall enough
                    ax.text(i, pos, str(value), ha='center', va='center', fontweight='bold', 
                        color='white' if category in ['Pre-assigned'] else 'black',
                        fontsize=8 if value < 5 else 10)
        bottom_tracker += np.array(data[category])
    
    # Mark restricted hobbies if available
    if restricted_hobbies:
        restricted_indices = [i for i, h in enumerate(hobbies) if h in restricted_hobbies]
        for i in restricted_indices:
            ax.text(i, -2, "*", ha='center', va='top', fontsize=24, color='red')
            
        # Add note about restricted hobbies
        if restricted_indices:
            ax.text(0.01, 0.01, "* Restricted hobbies (only pre-assigned campers)", transform=ax.transAxes, 
                   fontsize=10, color='red', ha='left')
    
    # Set axis labels and title
    ax.set_xlabel('Hobby', fontsize=12)
    ax.set_ylabel('Number of Campers', fontsize=12)
    ax.set_title('Choice Distribution by Hobby', fontsize=16)
    
    # Set x-axis tick labels with rotation for better readability
    ax.set_xticks(range(len(hobbies)))
    ax.set_xticklabels(hobbies, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for rotated labels
    plt.tight_layout()
    
    return fig

# Initialize session state variables
def init_session_state():
    if 'has_run' not in st.session_state:
        st.session_state.has_run = False
    if 'allocator' not in st.session_state:
        st.session_state.allocator = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()

# Perform initialization
if not check_dependencies():
    st.stop()

init_session_state()

# Update last activity time
st.session_state.last_activity = datetime.now()

# Main app title
st.title("Camp Northland Hobby Allocator")
st.write("This application automates the allocation of campers to hobby activities based on their preferences.")

with st.expander("How to use this application", expanded=True):
    st.markdown("""
    1. **Upload the required CSV files** (camper preferences and hobby configuration)
    2. **Upload optional files** if needed (pre-assignments and previous allocations)
    3. **Adjust parameters** if desired (or leave as default)
    4. **Click 'Run Allocation'** and wait for processing to complete
    5. **View results and download files** when finished
    
    **Important**: All generated files are temporary and will be lost when you close this page.
    Make sure to download any files you want to keep!
    """)

# Add privacy notice
with st.expander("Privacy Information", expanded=False):
    st.markdown("""
    ### Privacy Policy
    - All data processing happens in your session
    - For maximum privacy, download your results as soon as they're generated
    - No data is shared with third parties
    - All data is automatically deleted when you close this page
    """)
    
    # Add clear data button
    if st.button("Clear All Data", key="clear_all_data"):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key not in ['last_activity']:
                del st.session_state[key]
        init_session_state()
        st.success("All data has been cleared from your session")
        st.rerun()

# Only show the input form if allocation hasn't been run yet
if not st.session_state.has_run:
    # Main content with equal columns for required and optional files
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Required Files")
        
        # Initialize hobby lists
        premium_hobbies_list = []
        restricted_hobbies_list = []

        input_file = st.file_uploader("Camper Preferences CSV", type=["csv"], 
                                   help="CSV with camper names and their hobby choices")
        
        if input_file:
            # Preview and validate the data
            try:
                input_df = pd.read_csv(input_file)
                input_file.seek(0)  # Reset file pointer after reading
                
                # Show row count
                st.success(f"✅ Camper preferences uploaded: {len(input_df)} campers")
                
                # Show preview of first 5 rows
                if st.checkbox("Preview camper data"):
                    st.dataframe(input_df.head(5))
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
            
        config_file = st.file_uploader("Hobby Configuration CSV", type=["csv"], 
                                    help="CSV with hobby details including capacities")
        
        if config_file and input_file:
            # Try to read the config file to extract premium and restricted activities
            try:
                config_df = pd.read_csv(config_file)
                config_file.seek(0)  # Reset file pointer after reading
                
                # Look for the Hobby Name column (could be named differently)
                hobby_col = None
                for col in config_df.columns:
                    if 'hobby' in col.lower() and 'name' in col.lower():
                        hobby_col = col
                        break
                if not hobby_col and 'Name' in config_df.columns:
                    hobby_col = 'Name'
                
                # Show preview of config
                if st.checkbox("Preview hobby configuration"):
                    st.dataframe(config_df.head())
                
                # Check if "Premium" column exists (case insensitive)
                premium_col = None
                for col in config_df.columns:
                    if col.lower() == 'premium':
                        premium_col = col
                        break
                
                if premium_col and hobby_col:
                    # Extract premium activities
                    premium_values = config_df[premium_col].astype(str).str.lower()
                    premium_hobbies = config_df[premium_values.isin(['yes', 'true', '1', 'y'])]
                    
                    if not premium_hobbies.empty:
                        premium_hobbies_list = premium_hobbies[hobby_col].tolist()
                        if premium_hobbies_list:
                            st.info(f"Found premium activities in config: {', '.join(premium_hobbies_list)}")
                
                # Check if "Restricted" column exists and show information
                restricted_col = None
                for col in config_df.columns:
                    if col.lower() == 'restricted':
                        restricted_col = col
                        break
                
                if restricted_col and hobby_col:
                    # Count restricted hobbies
                    restricted_values = config_df[restricted_col].astype(str).str.lower()
                    restricted_hobbies = config_df[restricted_values.isin(['yes', 'true', '1', 'y'])]
                    
                    if not restricted_hobbies.empty:
                        restricted_hobbies_list = restricted_hobbies[hobby_col].tolist()
                        if restricted_hobbies_list:
                            st.info(f"Found restricted hobbies in config: {', '.join(restricted_hobbies_list)}")
                else:
                    # Show info about the new restricted option if not present
                    st.info("Note: Your hobby configuration doesn't include a 'Restricted' column. Adding this column allows you to control which pre-assigned hobbies are exclusive.")
            except Exception as e:
                st.warning(f"Could not read hobby configuration file: {str(e)}")
                
            st.success("✅ Hobby configuration uploaded")

    with col2:
        st.header("Optional Files")
        pre_assignments_file = st.file_uploader("Pre-assignments CSV", type=["csv"], 
                                         help="CSV with pre-assigned campers (optional)")
        
        if pre_assignments_file:
            try:
                pre_df = pd.read_csv(pre_assignments_file)
                pre_assignments_file.seek(0)  # Reset file pointer
                st.success(f"✅ Pre-assignments uploaded: {len(pre_df)} campers")
                
                if st.checkbox("Preview pre-assignments"):
                    st.dataframe(pre_df.head())
            except Exception as e:
                st.error(f"Error reading pre-assignments file: {str(e)}")
        
        previous_allocations_file = st.file_uploader("Previous Allocations CSV", type=["csv"], 
                                             help="CSV with previous week's allocations (optional)")
        
        if previous_allocations_file:
            try:
                prev_df = pd.read_csv(previous_allocations_file)
                previous_allocations_file.seek(0)  # Reset file pointer
                st.success(f"✅ Previous allocations uploaded: {len(prev_df)} records")
                
                if st.checkbox("Preview previous allocations"):
                    st.dataframe(prev_df.head())
            except Exception as e:
                st.error(f"Error reading previous allocations file: {str(e)}")

    st.header("Parameters")
    col3, col4 = st.columns(2)

    with col3:
        # Default values from main.py (confirmed to be 0.2)
        weight_factor = st.slider("Priority Weight Factor", 0.0, 1.0, 0.2, 0.05, 
                              help="Higher values give more priority to campers who didn't get top choices previously")
        
        # Default values from main.py (confirmed to be 0.1)
        premium_factor = st.slider("Premium Activity Factor", 0.0, 1.0, 0.1, 0.05, 
                              help="Lower values make premium activities more exclusive to first-choice campers")

    with col4:
        # If we successfully read premium activities from config, display them as default
        default_premium = ", ".join(premium_hobbies_list) if premium_hobbies_list else ""
        premium_activities = st.text_input("Premium Activities (comma-separated)", 
                                      value=default_premium,
                                      help="Special activities that should prioritize first-choice campers (auto-populated from config file if available)")
        
        # Similar for restricted activities
        default_restricted = ", ".join(restricted_hobbies_list) if restricted_hobbies_list else ""
        restricted_activities = st.text_input("Restricted Activities (comma-separated)",
                                         value=default_restricted,
                                         help="Activities that only allow pre-assigned campers (auto-populated from config file if available)")

    # Run button
    if st.button("Run Allocation", key="run_allocation_button", type="primary", use_container_width=True):
        if not input_file or not config_file:
            st.error("Error: Both camper preferences and hobby configuration files are required.")
        else:
            # Create progress display
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                status_area = st.empty()
            
            try:
                # Initialize the allocator
                allocator = HobbyAllocator()
                
                # Save uploaded files to temporary locations for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f_input:
                    f_input.write(input_file.getvalue())
                    input_path = f_input.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f_config:
                    f_config.write(config_file.getvalue())
                    config_path = f_config.name
                
                pre_path = None
                if pre_assignments_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f_pre:
                        f_pre.write(pre_assignments_file.getvalue())
                        pre_path = f_pre.name
                        
                prev_path = None
                if previous_allocations_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f_prev:
                        f_prev.write(previous_allocations_file.getvalue())
                        prev_path = f_prev.name
                
                # Step 1: Read input data
                status_area.info("Reading input data...")
                progress_bar.progress(0.1)
                allocator.read_input_data(input_path)
                allocator.read_hobby_config(config_path)
                
                # Step 2: Read optional data
                progress_bar.progress(0.2)
                if pre_path:
                    allocator.read_pre_assignments(pre_path)
                if prev_path:
                    allocator.read_previous_allocations(prev_path, weight_factor)
                
                # Step 3: Set up premium and restricted activities
                progress_bar.progress(0.3)
                premium_activities_list = None
                if premium_activities:
                    premium_activities_list = [activity.strip() for activity in premium_activities.split(',')]
                
                if restricted_activities:
                    restricted_activities_list = [activity.strip() for activity in restricted_activities.split(',')]
                    allocator.restricted_hobbies = restricted_activities_list
                
                # Step 4: Run optimization
                status_area.info("Running optimization... This may take a few minutes.")
                progress_bar.progress(0.4)
                
                allocator.create_allocation_model(
                    premium_activities=premium_activities_list, 
                    premium_factor=premium_factor
                )
                
                progress_bar.progress(0.8)
                status_area.success("Optimization complete! Generating reports...")
                
                # Step 5: Generate all results in memory
                # Create summary
                summary_data = []
                hobby_counts = {}
                pre_assigned_counts = {}
                
                for i, hobby in allocator.allocations.items():
                    hobby_counts[hobby] = hobby_counts.get(hobby, 0) + 1
                    if allocator.pre_assignments and i in allocator.pre_assignments:
                        pre_assigned_counts[hobby] = pre_assigned_counts.get(hobby, 0) + 1
                
                for _, row in allocator.hobby_config.iterrows():
                    hobby_name = row['Name']
                    assigned = hobby_counts.get(hobby_name, 0)
                    pre_assigned = pre_assigned_counts.get(hobby_name, 0)
                    
                    location = row.get('Location', 'Unknown')
                    leader = row.get('Leader', row.get('Specialty', 'Unknown'))
                    allowed_groups = row.get('Allowed Groups', row.get('Allowed Divisions', 'All'))
                    
                    min_capacity = row['Min Capacity']
                    max_capacity = row['Max Capacity']
                    
                    is_restricted = "No"
                    if 'Restricted' in row:
                        restricted_value = str(row['Restricted']).lower()
                        is_restricted = "Yes" if restricted_value in ['yes', 'true', '1', 'y'] else "No"
                    else:
                        is_restricted = "Yes" if hobby_name in allocator.restricted_hobbies else "No"
                    
                    summary_data.append({
                        'Hobby Name': hobby_name,
                        'Location': location,
                        'Hobby Leader': leader,
                        'Allowed Age Groups': allowed_groups,
                        'Min Capacity': min_capacity,
                        'Max Capacity': max_capacity,
                        'Number of Campers Assigned': assigned,
                        'Pre-assigned Campers': pre_assigned,
                        'Regular Assigned Campers': assigned - pre_assigned,
                        'Available Slots': max(0, max_capacity - assigned),
                        'Restricted': is_restricted
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Create master DataFrame
                master_df = allocator.campers_df.copy()
                master_df['Assigned Hobby'] = None
                master_df['Choice Rank'] = None
                master_df['Pre-assigned'] = 'No'
                master_df['Priority Weight'] = 1.0
                master_df['Previous Choice Rank'] = None
                
                for i, hobby in allocator.allocations.items():
                    master_df.loc[i, 'Assigned Hobby'] = hobby
                    
                    if allocator.pre_assignments and i in allocator.pre_assignments:
                        master_df.loc[i, 'Pre-assigned'] = 'Yes'
                    
                    if hasattr(allocator, 'weights') and allocator.weights and i in allocator.weights:
                        master_df.loc[i, 'Priority Weight'] = allocator.weights[i]
                    
                    if hasattr(allocator, 'previous_allocations') and allocator.previous_allocations and i in allocator.previous_allocations:
                        master_df.loc[i, 'Previous Choice Rank'] = allocator.previous_allocations[i]
                    
                    choice_rank = 0
                    for j in range(1, 6):
                        choice_col = f'Choice {j}'
                        if choice_col in master_df.columns and master_df.loc[i, choice_col] == hobby:
                            choice_rank = j
                            break
                    master_df.loc[i, 'Choice Rank'] = choice_rank
                
                # Create choice distribution data
                choice_data = {}
                for i in range(6):  # 0-5 for no match, 1st-5th choice
                    choice_data[i] = 0
                choice_data['pre'] = 0
                
                # Create hobby choice counts data
                hobby_choice_counts = {}
                hobbies = allocator.hobby_config['Name'].tolist()
                for hobby in hobbies:
                    hobby_choice_counts[hobby] = {
                        1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 0: 0, 'pre': 0
                    }
                
                for i, hobby in allocator.allocations.items():
                    if allocator.pre_assignments and i in allocator.pre_assignments:
                        choice_data['pre'] += 1
                        hobby_choice_counts[hobby]['pre'] += 1
                        continue
                    
                    choice_rank = 0
                    for j in range(1, 6):
                        choice_col = f'Choice {j}'
                        if choice_col in allocator.campers_df.columns and pd.notna(allocator.campers_df.loc[i, choice_col]) and allocator.campers_df.loc[i, choice_col] == hobby:
                            choice_rank = j
                            break
                    
                    choice_data[choice_rank] += 1
                    hobby_choice_counts[hobby][choice_rank] += 1
                
                # Create visualizations
                progress_bar.progress(0.9)
                status_area.info("Creating visualizations...")
                
                allocation_chart = create_allocation_bar_chart(summary_df)
                choice_chart = create_choice_distribution_chart(choice_data)
                hobby_choice_chart = create_hobby_choice_distribution_chart(hobby_choice_counts, allocator.restricted_hobbies)
                
                # Save chart images to bytes
                allocation_img_bytes = io.BytesIO()
                allocation_chart.savefig(allocation_img_bytes, format='png', dpi=300, bbox_inches='tight')
                allocation_img_bytes.seek(0)
                
                choice_img_bytes = io.BytesIO()
                choice_chart.savefig(choice_img_bytes, format='png', dpi=300, bbox_inches='tight')
                choice_img_bytes.seek(0)
                
                hobby_choice_img_bytes = io.BytesIO()
                hobby_choice_chart.savefig(hobby_choice_img_bytes, format='png', dpi=300, bbox_inches='tight')
                hobby_choice_img_bytes.seek(0)
                
                # Store results in session state
                st.session_state.results = {
                    'allocator': allocator,
                    'summary_df': summary_df,
                    'master_df': master_df,
                    'choice_data': choice_data,
                    'hobby_choice_counts': hobby_choice_counts,
                    'allocation_chart': allocation_chart,
                    'choice_chart': choice_chart,
                    'hobby_choice_chart': hobby_choice_chart,
                    'allocation_img_bytes': allocation_img_bytes.getvalue(),
                    'choice_img_bytes': choice_img_bytes.getvalue(),
                    'hobby_choice_img_bytes': hobby_choice_img_bytes.getvalue()
                }
                
                # Clean up temporary files
                for file_path in [input_path, config_path, pre_path, prev_path]:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                
                progress_bar.progress(1.0)
                status_area.success("Allocation completed successfully!")
                
                # Set completion flag
                st.session_state.has_run = True
                
                # Clear progress display and rerun to show results
                progress_placeholder.empty()
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred during allocation: {str(e)}")
                st.error("Please check your input files and try again.")
                
                # Clean up temporary files on error
                for file_path in [input_path, config_path, pre_path, prev_path]:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.unlink(file_path)
                        except:
                            pass

else:
    # Display results section
    st.success("✅ Allocation completed successfully!")
    
    # Add Start New Allocation button at the top of results
    if st.button("Start New Allocation", key="results_new_allocation", use_container_width=True):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key not in ['last_activity']:
                del st.session_state[key]
        init_session_state()
        st.rerun()
    
    # Get results from session state
    results = st.session_state.results
    allocator = results['allocator']
    summary_df = results['summary_df']
    master_df = results['master_df']
    choice_data = results['choice_data']
    
    # Create a results section
    st.header("Allocation Results")
    
    # Calculate satisfaction score (sum of top 3 choices)
    first_count = choice_data.get(1, 0)
    second_count = choice_data.get(2, 0)
    third_count = choice_data.get(3, 0)
    
    total_regular = sum(choice_data[i] for i in range(6))  # 0-5
    satisfaction_score = (first_count + second_count + third_count) / total_regular * 100 if total_regular > 0 else 0
    
    # Display metrics
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Overall Satisfaction", f"{satisfaction_score:.1f}%", 
                        help="Percentage of campers assigned to their top 3 choices")
    
    if first_count > 0:
        first_pct = (first_count / total_regular) * 100 if total_regular > 0 else 0
        metrics_cols[1].metric("First Choice", f"{first_pct:.1f}%", 
                            help=f"{first_count} campers")
    
    if second_count > 0:
        second_pct = (second_count / total_regular) * 100 if total_regular > 0 else 0
        metrics_cols[2].metric("Second Choice", f"{second_pct:.1f}%", 
                            help=f"{second_count} campers")
    
    if third_count > 0:
        third_pct = (third_count / total_regular) * 100 if total_regular > 0 else 0
        metrics_cols[3].metric("Third Choice", f"{third_pct:.1f}%", 
                            help=f"{third_count} campers")
    
    # Additional metrics row
    if choice_data:
        metrics_row2 = st.columns(4)
        idx = 0
        
        for choice in [4, 5, 0, 'pre']:
            if choice in choice_data and choice_data[choice] > 0:
                count = choice_data[choice]
                if choice == 0:
                    label = "No Match"
                    pct = (count / total_regular) * 100 if total_regular > 0 else 0
                elif choice == 'pre':
                    label = "Pre-assigned"
                    total_with_pre = total_regular + count
                    pct = (count / total_with_pre) * 100 if total_with_pre > 0 else 0
                else:
                    label = f"{choice_to_word(choice)} Choice"
                    pct = (count / total_regular) * 100 if total_regular > 0 else 0
                    
                metrics_row2[idx].metric(label, f"{pct:.1f}%", 
                                    help=f"{count} campers")
                idx += 1
                if idx >= 4:  # Only show up to 4 metrics in this row
                    break
    
    # Display visualizations
    st.subheader("Visualizations")
    
    # Create tabs for visualizations
    tab1, tab2, tab3 = st.tabs(["Hobby Allocation", "Choice Distribution", "Choice Distribution by Hobby"])
    
    with tab1:
        st.pyplot(results['allocation_chart'])
        st.download_button(
            label="Download Allocation Chart",
            data=results['allocation_img_bytes'],
            file_name="allocation_chart.png",
            mime="image/png",
            use_container_width=True
        )
    
    with tab2:
        st.pyplot(results['choice_chart'])
        st.download_button(
            label="Download Choice Distribution Chart",
            data=results['choice_img_bytes'],
            file_name="choice_distribution.png",
            mime="image/png",
            use_container_width=True
        )
    
    with tab3:
        st.pyplot(results['hobby_choice_chart'])
        st.download_button(
            label="Download Hobby Choice Distribution Chart",
            data=results['hobby_choice_img_bytes'],
            file_name="hobby_choice_distribution.png",
            mime="image/png",
            use_container_width=True
        )
    
    # Display summary data
    st.subheader("Allocation Summary")
    
    if 'Restricted' in summary_df.columns:
        st.info("Note: 'Restricted' hobbies only allow pre-assigned campers. " +
              "Non-restricted hobbies adjust their capacity based on pre-assignments.")
    
    st.dataframe(summary_df, use_container_width=True)
    
    # Display choice distribution stats in a table
    st.subheader("Choice Distribution")
    choice_data_table = []
    
    # Order the choices correctly
    ordered_choices = [1, 2, 3, 4, 5, 0, 'pre']
    
    for choice in ordered_choices:
        if choice in choice_data and choice_data[choice] > 0:
            count = choice_data[choice]
            
            if choice == 0:
                label = "No Match"
                percentage = (count / total_regular) * 100 if total_regular > 0 else 0
            elif choice == 'pre':
                label = "Pre-assigned"
                total_with_pre = total_regular + count
                percentage = (count / total_with_pre) * 100 if total_with_pre > 0 else 0
            else:
                label = choice_to_word(choice) + " Choice"
                percentage = (count / total_regular) * 100 if total_regular > 0 else 0
                
            choice_data_table.append({
                "Choice": label,
                "Campers": count,
                "Percentage": f"{percentage:.1f}%"
            })
    
    # Create dataframe and display
    if choice_data_table:
        choice_df = pd.DataFrame(choice_data_table)
        st.dataframe(choice_df, use_container_width=True)
    
    # Download section
    st.subheader("Download Files")
    
    # Prepare files for download
    files_to_zip = {
        'allocation_summary.csv': summary_df,
        'master_allocation.xlsx': master_df,
        'allocation_chart.png': results['allocation_img_bytes'],
        'choice_distribution.png': results['choice_img_bytes']
    }
    
    # Create hobby allocation Excel file
    hobby_dfs = {}
    for camper_idx, hobby in allocator.allocations.items():
        if hobby not in hobby_dfs:
            hobby_dfs[hobby] = []
        
        camper = allocator.campers_df.iloc[camper_idx]
        cabin = camper.get('Cabin', 'Unknown') if pd.notna(camper.get('Cabin', '')) else "Unknown"
        pre_assigned = allocator.pre_assignments and camper_idx in allocator.pre_assignments
        
        hobby_dfs[hobby].append({
            'Name': camper['Full Name'],
            'Division': camper['Division'],
            'Cabin': cabin,
            'Pre-assigned': 'Yes' if pre_assigned else 'No'
        })
    
    # Create division allocation Excel file
    division_dfs = {}
    for camper_idx, hobby in allocator.allocations.items():
        camper = allocator.campers_df.iloc[camper_idx]
        division = camper['Division']
        
        if division not in division_dfs:
            division_dfs[division] = []
        
        cabin = camper.get('Cabin', 'Unknown') if pd.notna(camper.get('Cabin', '')) else "Unknown"
        pre_assigned = allocator.pre_assignments and camper_idx in allocator.pre_assignments
        
        division_dfs[division].append({
            'Cabin': cabin,
            'Name': camper['Full Name'],
            'Assigned Hobby': hobby,
            'Pre-assigned': 'Yes' if pre_assigned else 'No'
        })
    
    # Create download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Download individual files
        st.subheader("Individual Files")
        
        # Summary CSV
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Summary (CSV)",
            data=csv_buffer.getvalue(),
            file_name="allocation_summary.csv",
            mime="text/csv"
        )
        
        # Master Excel
        excel_buffer = io.BytesIO()
        master_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        st.download_button(
            label="Download Master Allocation (Excel)",
            data=excel_buffer.getvalue(),
            file_name="master_allocation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Hobby allocation Excel
        hobby_excel_buffer = io.BytesIO()
        with pd.ExcelWriter(hobby_excel_buffer, engine='openpyxl') as writer:
            for hobby, data in hobby_dfs.items():
                if data:
                    df = pd.DataFrame(data).sort_values(by=['Division', 'Name'])
                    sheet_name = str(hobby)[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        st.download_button(
            label="Download Hobby Allocation (Excel)",
            data=hobby_excel_buffer.getvalue(),
            file_name="hobby_allocation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Division allocation Excel
        division_excel_buffer = io.BytesIO()
        with pd.ExcelWriter(division_excel_buffer, engine='openpyxl') as writer:
            for division, data in division_dfs.items():
                if data:
                    df = pd.DataFrame(data).sort_values(by=['Cabin', 'Name'])
                    sheet_name = str(division)[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        st.download_button(
            label="Download Division Allocation (Excel)",
            data=division_excel_buffer.getvalue(),
            file_name="division_allocation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.subheader("All Files")
        
        # Create comprehensive ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add summary CSV
            csv_data = io.StringIO()
            summary_df.to_csv(csv_data, index=False)
            zip_file.writestr('allocation_summary.csv', csv_data.getvalue())
            
            # Add master Excel
            excel_data = io.BytesIO()
            master_df.to_excel(excel_data, index=False, engine='openpyxl')
            zip_file.writestr('master_allocation.xlsx', excel_data.getvalue())
            
            # Add hobby allocation Excel
            hobby_excel_data = io.BytesIO()
            with pd.ExcelWriter(hobby_excel_data, engine='openpyxl') as writer:
                for hobby, data in hobby_dfs.items():
                    if data:
                        df = pd.DataFrame(data).sort_values(by=['Division', 'Name'])
                        sheet_name = str(hobby)[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            zip_file.writestr('hobby_allocation.xlsx', hobby_excel_data.getvalue())
            
            # Add division allocation Excel
            division_excel_data = io.BytesIO()
            with pd.ExcelWriter(division_excel_data, engine='openpyxl') as writer:
                for division, data in division_dfs.items():
                    if data:
                        df = pd.DataFrame(data).sort_values(by=['Cabin', 'Name'])
                        sheet_name = str(division)[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            zip_file.writestr('division_allocation.xlsx', division_excel_data.getvalue())
            
            # Add visualizations
            zip_file.writestr('allocation_chart.png', results['allocation_img_bytes'])
            zip_file.writestr('choice_distribution.png', results['choice_img_bytes'])
            zip_file.writestr('hobby_choice_distribution.png', results['hobby_choice_img_bytes'])
            
            # Add next week's previous allocations
            next_week_df = pd.DataFrame()
            next_week_df['Full Name'] = allocator.campers_df['Full Name']
            next_week_df['Division'] = allocator.campers_df['Division']
            next_week_df['Assigned Hobby'] = None
            next_week_df['Choice Rank'] = None
            
            for i, hobby in allocator.allocations.items():
                next_week_df.loc[i, 'Assigned Hobby'] = hobby
                choice_rank = 0
                for j in range(1, 6):
                    choice_col = f'Choice {j}'
                    if choice_col in allocator.campers_df.columns and allocator.campers_df.loc[i, choice_col] == hobby:
                        choice_rank = j
                        break
                next_week_df.loc[i, 'Choice Rank'] = choice_rank
            
            next_week_csv = io.StringIO()
            next_week_df.to_csv(next_week_csv, index=False)
            zip_file.writestr('next_week_previous_allocations.csv', next_week_csv.getvalue())
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📦 Download All Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"hobby_allocation_complete_{timestamp}.zip",
            mime="application/zip",
            use_container_width=True
        )

# Add help information to sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This is the Camp Northland Hobby Allocator tool. "
    "It helps automatically assign campers to their preferred hobby activities "
    "while optimizing overall satisfaction and meeting constraints."
)

# Add system status check to sidebar
with st.sidebar.expander("System Status", expanded=False):
    st.text(f"App Version: {APP_VERSION}")
    st.text(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    st.text("✅ Integrated Architecture")
    st.text("✅ No Subprocess Dependencies")

st.sidebar.title("Help")
st.sidebar.markdown(
    """
    **Required Files Format:**
    - **Camper Preferences**: CSV with camper names and their hobby choices  
    - **Hobby Configuration**: CSV with hobby details and capacities
    
    **Pre-Assignment Options:**
    - Add a 'Restricted' column to your hobby config file
    - Set to 'Yes' for hobbies that only allow pre-assigned campers
    - Set to 'No' for hobbies that adjust capacity based on pre-assignments
    
    **Output Files:**
    - All files are created temporarily during your session
    - You must download files to save them
    - No data is saved permanently
    
    **Questions or Issues?**
    Contact the program administrator for assistance.
    """
)