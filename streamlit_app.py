import streamlit as st
import os
import subprocess
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
APP_VERSION = "1.0.0"

# Add local bin to PATH for Streamlit Cloud
if '/home/appuser/.local/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/home/appuser/.local/bin:' + os.environ.get('PATH', '')

# First, check for required dependencies
def check_pulp_installation():
    """Comprehensive check of PuLP installation and solver availability"""
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
            return True
        else:
            st.sidebar.error("❌ No solvers available - optimization will fail")
            return False
            
    except ImportError as e:
        st.sidebar.error(f"❌ PuLP import failed: {e}")
        st.error("""
        ### Error: PuLP library not found
        
        This application requires the PuLP optimization library, which seems to be missing.
        
        If you're running this locally, please install it with:
        ```
        pip install pulp==2.7.0
        ```
        
        If you're seeing this on Streamlit Cloud, please contact the administrator.
        """)
        return False

# Set page configuration
st.set_page_config(
    page_title="Camp Northland Hobby Allocator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Function to list all PNG files in a directory
def find_visualization_files(directory):
    """Find all PNG files in the given directory"""
    # Use glob to recursively search for PNG files
    pattern = os.path.join(directory, "**/*.png")
    image_files = glob.glob(pattern, recursive=True)
    return image_files

# Function to create a zip file containing all files
def create_zip_of_files(files):
    """Create a zip file in memory containing all the specified files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files:
            # Get relative path for cleaner zip structure
            arcname = os.path.basename(file_path)
            zip_file.write(file_path, arcname=arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Extract choice counts from output log using regex
def extract_choice_data_from_log(output_lines):
    """Parse the detailed output log to extract accurate choice distribution data"""
    choice_data = {
        1: (0, 0),  # (count, percentage)
        2: (0, 0),
        3: (0, 0),
        4: (0, 0),
        5: (0, 0),
        0: (0, 0),  # No match
        'pre': (0, 0)  # Pre-assigned
    }
    
    # Patterns to match different lines in the output
    first_choice_pattern = r"First choice: (\d+) campers \(([0-9.]+)%\)"
    second_choice_pattern = r"Second choice: (\d+) campers \(([0-9.]+)%\)"
    third_choice_pattern = r"Third choice: (\d+) campers \(([0-9.]+)%\)"
    fourth_choice_pattern = r"Fourth choice: (\d+) campers \(([0-9.]+)%\)"
    fifth_choice_pattern = r"Fifth choice: (\d+) campers \(([0-9.]+)%\)"
    no_match_pattern = r"No choice matched: (\d+) campers \(([0-9.]+)%\)"
    
    # Try to find the statements in the logs
    for line in output_lines:
        # First choice
        match = re.search(first_choice_pattern, line)
        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            choice_data[1] = (count, percentage)
            continue
            
        # Second choice
        match = re.search(second_choice_pattern, line)
        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            choice_data[2] = (count, percentage)
            continue
            
        # Third choice
        match = re.search(third_choice_pattern, line)
        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            choice_data[3] = (count, percentage)
            continue
            
        # Fourth choice
        match = re.search(fourth_choice_pattern, line)
        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            choice_data[4] = (count, percentage)
            continue
            
        # Fifth choice
        match = re.search(fifth_choice_pattern, line)
        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            choice_data[5] = (count, percentage)
            continue
            
        # No match
        match = re.search(no_match_pattern, line)
        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            choice_data[0] = (count, percentage)
            continue
        
        # Check if the line contains pre-assigned count
        if "Pre-assigned campers" in line and ": " in line:
            parts = line.split(": ")
            if len(parts) > 1:
                try:
                    pre_count = int(parts[1].strip())
                    # Calculate percentage based on sum of all other counts
                    total = sum(count for choice, (count, _) in choice_data.items() if choice != 'pre')
                    if total > 0:
                        pre_percentage = (pre_count / (total + pre_count)) * 100
                    else:
                        pre_percentage = 0
                    choice_data['pre'] = (pre_count, pre_percentage)
                except ValueError:
                    pass
    
    return choice_data

# Read choice distribution from master Excel file
def read_choice_distribution_from_master(output_dir):
    """Extract choice distribution data from the master allocation file"""
    master_path = os.path.join(output_dir, "master_allocation.xlsx")
    if not os.path.exists(master_path):
        return None
        
    try:
        df = pd.read_excel(master_path)
        
        # Check if required columns exist
        if 'Choice Rank' not in df.columns:
            return None
            
        # Count values
        choice_counts = df['Choice Rank'].value_counts().to_dict()
        
        # Check if Pre-assigned column exists
        pre_count = 0
        if 'Pre-assigned' in df.columns:
            pre_count = (df['Pre-assigned'] == 'Yes').sum()
            
        # Calculate percentages
        total = len(df)
        choice_data = {}
        
        for choice in range(1, 6):
            count = choice_counts.get(choice, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            choice_data[choice] = (count, percentage)
            
        # No match (0)
        count = choice_counts.get(0, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        choice_data[0] = (count, percentage)
        
        # Pre-assigned
        percentage = (pre_count / total) * 100 if total > 0 else 0
        choice_data['pre'] = (pre_count, percentage)
        
        return choice_data
    except Exception as e:
        st.error(f"Error reading master file: {str(e)}")
        return None

# Direct plotting functions in case file loading fails
def create_allocation_bar_chart(summary_df):
    """Create an allocation bar chart directly in Streamlit"""
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    """Create a choice distribution chart directly in Streamlit"""
    # Process choice distribution data
    choices = ['First Choice', 'Second Choice', 'Third Choice', 'Fourth Choice', 'Fifth Choice', 'No Choice Match']
    counts = [
        choice_data[1][0],  # First choice
        choice_data[2][0],  # Second choice
        choice_data[3][0],  # Third choice
        choice_data[4][0],  # Fourth choice
        choice_data[5][0],  # Fifth choice
        choice_data[0][0],  # No choice match
    ]
    
    # Calculate percentages based on the total
    total_campers = sum(counts)
    if 'pre' in choice_data and choice_data['pre'][0] > 0:
        total_campers += choice_data['pre'][0]
        choices.append('Pre-assigned')
        counts.append(choice_data['pre'][0])
    
    percentages = [count / total_campers * 100 for count in counts] if total_campers > 0 else [0] * len(counts)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
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

# Check if the main_updated.py script exists
def check_required_files():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_updated.py")
    if not os.path.exists(script_path):
        st.error("Critical error: Required script 'main_updated.py' not found. Please ensure it's in the same directory as this app.")
        st.stop()

# Initialize session state for temporary directory management
def init_temp_dir():
    if 'temp_dir' not in st.session_state:
        # Create a temporary directory that will be automatically cleaned up
        st.session_state.temp_dir = tempfile.mkdtemp()
        
        # Register cleanup function to run when the app exits
        def cleanup_temp_files():
            if os.path.exists(st.session_state.temp_dir):
                try:
                    shutil.rmtree(st.session_state.temp_dir)
                except:
                    pass
        
        atexit.register(cleanup_temp_files)

# Initialize session state variables
def init_session_state():
    if 'has_run' not in st.session_state:
        st.session_state.has_run = False
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'visualization_files' not in st.session_state:
        st.session_state.visualization_files = []
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None
    if 'choice_distribution' not in st.session_state:
        st.session_state.choice_distribution = {}
    if 'satisfaction_score' not in st.session_state:
        st.session_state.satisfaction_score = None
    if 'all_files' not in st.session_state:
        st.session_state.all_files = []
    if 'output_log' not in st.session_state:
        st.session_state.output_log = []
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()

# Perform initialization
check_required_files()
init_temp_dir()
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
        # Keep only essential session state keys
        keys_to_keep = ['temp_dir', 'last_activity']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        # Reinitialize necessary state variables
        st.session_state.has_run = False
        st.session_state.visualization_files = []
        st.session_state.all_files = []
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

    # Create temp directory for outputs
    temp_output_dir = tempfile.mkdtemp(dir=st.session_state.temp_dir)

    # Run button
    if st.button("Run Allocation", key="run_allocation_button", type="primary", use_container_width=True):
        if not input_file or not config_file:
            st.error("Error: Both camper preferences and hobby configuration files are required.")
        else:
            # Check PuLP availability before proceeding
            if not check_pulp_installation():
                st.stop()
            
            # Create progress display
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                status_area = st.empty()
            
            output_area = st.expander("Detailed Output Log", expanded=False)
            
            # Create unique output directory within our temp directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(temp_output_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            # Store the output directory in session state
            st.session_state.output_dir = output_dir
            
            # Save uploaded files to temporary locations
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
            
            # Get the path to the main_updated.py script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            main_script_path = os.path.join(script_dir, "main_updated.py")
            
            # Build command
            cmd = ["python", main_script_path, 
                   "--input", input_path,
                   "--config", config_path,
                   "--weight-factor", str(weight_factor),
                   "--premium-factor", str(premium_factor),
                   "--output-dir", output_dir]
            
            if pre_path:
                cmd.extend(["--pre-assignments", pre_path])
            if prev_path:
                cmd.extend(["--previous-allocations", prev_path])
            
            # Add premium activities if specified (override config)
            if premium_activities:
                cmd.extend(["--premium-activities", premium_activities])
                
            # Add restricted activities as a command-line argument if specified
            if restricted_activities:
                cmd.extend(["--restricted-activities", restricted_activities])
            
            status_area.info("Running allocation... Please wait, this may take a few minutes.")
            output_area.text(f"Command: {' '.join(cmd)}")
            
            try:
                # Run allocation
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                output_lines = []
                
                for i, line in enumerate(process.stdout):
                    line_text = line.strip()
                    output_lines.append(line_text)
                    output_area.text("\n".join(output_lines[-50:]))  # Show only last 50 lines for performance
                    
                    # Update progress (approximate)
                    if "Model solved with satisfaction score" in line_text:
                        progress_bar.progress(0.8)
                        status_area.success("Optimization complete! Generating reports...")
                    elif "Creating" in line_text and "visualization" in line_text:
                        progress_bar.progress(0.9)
                    elif i % 10 == 0:  # Update progress periodically
                        progress_bar.progress(min(0.1 + i/500, 0.7))  # Cap at 70% until we know it's done
                
                process.wait()
                
                # Store the output log in session state
                st.session_state.output_log = output_lines
                
                # Show results
                if process.returncode == 0:
                    # Clear the progress display
                    progress_placeholder.empty()
                    
                    # Set the has_run flag to true
                    st.session_state.has_run = True
                    
                    # Extract choice data from output log
                    choice_data = extract_choice_data_from_log(output_lines)
                    
                    # Store in session state
                    st.session_state.choice_distribution = choice_data
                    
                    # Get satisfaction score
                    satisfaction_score = None
                    for line in output_lines:
                        if "Satisfaction score:" in line:
                            try:
                                satisfaction_score = float(line.split(":")[-1].strip().rstrip("%"))
                            except:
                                pass
                    
                    st.session_state.satisfaction_score = satisfaction_score
                    
                    # Wait a moment for files to be completely written to disk
                    time.sleep(2)
                    
                    # Find all files in the output directory
                    all_files = []
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            all_files.append(file_path)
                    
                    st.session_state.all_files = all_files
                    
                    # Find visualization files with recursive glob
                    image_files = find_visualization_files(output_dir)
                    st.session_state.visualization_files = image_files
                    
                    # Try to read summary file
                    summary_path = os.path.join(output_dir, "allocation_summary.csv")
                    summary_df = None
                    if os.path.exists(summary_path):
                        try:
                            summary_df = pd.read_csv(summary_path)
                            st.session_state.summary_df = summary_df
                        except Exception as e:
                            st.warning(f"Could not read summary file: {str(e)}")
                    
                    # Rerun to display results page
                    st.rerun()
                else:
                    progress_bar.progress(1.0)
                    status_area.error(f"Error: Process exited with code {process.returncode}")
                    st.error("\n".join(output_lines[-10:]))  # Show last few lines of output
            
            except Exception as e:
                progress_bar.progress(1.0)
                status_area.error(f"Error during allocation process: {str(e)}")
                st.error(f"An error occurred during the allocation process: {str(e)}")
                st.error("Please check your input files and try again.")
            
            finally:
                # Clean up temporary files
                for file_path in [input_path, config_path]:
                    if file_path and os.path.exists(file_path):
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                
                for file_path in [pre_path, prev_path]:
                    if file_path and file_path is not None and os.path.exists(file_path):
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
        keys_to_keep = ['temp_dir', 'last_activity']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state.has_run = False
        st.rerun()
    
    # Create a results section
    st.header("Allocation Results")
    
    # Get choice data from session state
    choice_data = st.session_state.choice_distribution
    
    # Calculate satisfaction score (sum of top 3 choices)
    if choice_data and all(choice in choice_data for choice in [1, 2, 3]):
        # Get counts for first, second, and third choices
        first_count = choice_data[1][0] if 1 in choice_data else 0
        second_count = choice_data[2][0] if 2 in choice_data else 0
        third_count = choice_data[3][0] if 3 in choice_data else 0
        
        # Calculate total excluding pre-assigned
        total_regular = 0
        for choice, (count, _) in choice_data.items():
            if choice != 'pre':
                total_regular += count
        
        # Calculate satisfaction percentage
        if total_regular > 0:
            satisfaction_score = (first_count + second_count + third_count) / total_regular * 100
        else:
            satisfaction_score = 0
    else:
        # Fall back to the one extracted from logs
        satisfaction_score = st.session_state.satisfaction_score

    # Display metrics
    metrics_cols = st.columns(4)
    if satisfaction_score is not None:
        metrics_cols[0].metric("Overall Satisfaction", f"{satisfaction_score:.1f}%", 
                            help="Percentage of campers assigned to their top 3 choices")
    
    if choice_data and 1 in choice_data:
        first_count, first_pct = choice_data[1]
        metrics_cols[1].metric("First Choice", f"{first_pct:.1f}%", 
                            help=f"{first_count} campers")
    
    if choice_data and 2 in choice_data:
        second_count, second_pct = choice_data[2]
        metrics_cols[2].metric("Second Choice", f"{second_pct:.1f}%", 
                            help=f"{second_count} campers")
    
    if choice_data and 3 in choice_data:
        third_count, third_pct = choice_data[3]
        metrics_cols[3].metric("Third Choice", f"{third_pct:.1f}%", 
                            help=f"{third_count} campers")
    
    # Additional metrics row
    if choice_data and len(choice_data) > 3:
        metrics_row2 = st.columns(4)
        idx = 0
        
        for choice in [4, 5, 0, 'pre']:
            if choice in choice_data and choice_data[choice][0] > 0:
                count, pct = choice_data[choice]
                if choice == 0:
                    label = "No Match"
                elif choice == 'pre':
                    label = "Pre-assigned"
                else:
                    label = f"{choice_to_word(choice)} Choice"
                    
                metrics_row2[idx].metric(label, f"{pct:.1f}%", 
                                    help=f"{count} campers")
                idx += 1
                if idx >= 4:  # Only show up to 4 metrics in this row
                    break
    
    # Display visualizations
    st.subheader("Visualizations")
    
    # Create a toggle/selector for the visualizations
    image_files = st.session_state.visualization_files
    
    if image_files:
        # Create a dictionary of visualization types
        viz_types = {}
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            if "choice_distribution.png" == filename and "hobby" not in filename:
                viz_types["Choice Distribution"] = img_path
            elif "allocation_chart.png" == filename:
                viz_types["Hobby Allocation"] = img_path
            elif "hobby_choice_distribution.png" == filename:
                viz_types["Choice Distribution by Hobby"] = img_path
            else:
                # Add any other images
                viz_types[filename] = img_path
        
        # Create tabs for the visualizations
        if viz_types:
            tab_names = list(viz_types.keys())
            tabs = st.tabs(tab_names)
            
            for i, (name, img_path) in enumerate(viz_types.items()):
                with tabs[i]:
                    try:
                        # Read the image file
                        with open(img_path, "rb") as f:
                            image_bytes = f.read()
                            st.image(image_bytes, use_container_width=True)
                            
                            # Add download button below each image
                            st.download_button(
                                label=f"Download this visualization",
                                data=image_bytes,
                                file_name=os.path.basename(img_path),
                                mime="image/png",
                                key=f"download_viz_{i}",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error displaying image {name}: {str(e)}")
    else:
        # If no images were found, try to create them directly
        st.warning("No visualization files were found. Creating visualizations directly...")
        
        # Create choice distribution chart
        if choice_data:
            st.subheader("Choice Distribution")
            try:
                fig = create_choice_distribution_chart(choice_data)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating choice distribution chart: {str(e)}")
        
        # Create allocation chart if summary data is available
        if st.session_state.summary_df is not None:
            st.subheader("Hobby Allocation")
            try:
                fig = create_allocation_bar_chart(st.session_state.summary_df)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating allocation chart: {str(e)}")
    
    # Display summary file if available
    summary_df = st.session_state.summary_df
    if summary_df is not None:
        st.subheader("Allocation Summary")
        
        # Check if Restricted column exists
        if 'Restricted' in summary_df.columns:
            # Add info about what "Restricted" means
            st.info("Note: 'Restricted' hobbies only allow pre-assigned campers. " +
                  "Non-restricted hobbies adjust their capacity based on pre-assignments.")
        
        st.dataframe(summary_df, use_container_width=True)
    
    # Display choice distribution stats in a table
    st.subheader("Choice Distribution")
    choice_data_table = []
    
    # Order the choices correctly
    ordered_choices = [1, 2, 3, 4, 5, 0, 'pre']
    
    for choice in ordered_choices:
        if choice in choice_data and choice_data[choice][0] > 0:
            count, percentage = choice_data[choice]
            
            if choice == 0:
                label = "No Match"
            elif choice == 'pre':
                label = "Pre-assigned"
            else:
                label = choice_to_word(choice) + " Choice"
                
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
    
    # Add "Download All Files" button at the top of the download section
    if st.session_state.all_files:
        zip_data = create_zip_of_files(st.session_state.all_files)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📦 Download All Files as ZIP",
            data=zip_data,
            file_name=f"hobby_allocation_all_files_{timestamp}.zip",
            mime="application/zip",
            key="download_all_files",
            use_container_width=True
        )
    
    # Create mapping of file extensions to friendly names and MIME types
    file_types = {
        ".xlsx": ("Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ".csv": ("CSV", "text/csv"),
        ".png": ("Image", "image/png")
    }
    
    # Group files by type for better organization
    grouped_files = {}
    for file_path in st.session_state.all_files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext not in grouped_files:
            grouped_files[file_ext] = []
        
        grouped_files[file_ext].append((file_name, file_path))
    
    # Create a tab for each file type
    if grouped_files:
        file_tabs = []
        for ext, (name, _) in file_types.items():
            if ext in grouped_files:
                file_tabs.append(f"{name} Files ({len(grouped_files[ext])})")
            else:
                file_tabs.append(f"{name} Files (0)")
        
        if file_tabs:  # Only create tabs if there are files
            download_tabs = st.tabs(file_tabs)
            
            # Excel files
            with download_tabs[0]:
                if ".xlsx" in grouped_files:
                    cols = st.columns(3)
                    for i, (file_name, file_path) in enumerate(sorted(grouped_files[".xlsx"])):
                        try:
                            with open(file_path, "rb") as file:
                                file_data = file.read()
                                    
                            cols[i % 3].download_button(
                                label=file_name,
                                data=file_data,
                                file_name=file_name,
                                mime=file_types[".xlsx"][1],
                                key=f"xlsx_{i}",
                                use_container_width=True
                            )
                        except Exception as e:
                            cols[i % 3].error(f"Error with file {file_name}: {str(e)}")
                else:
                    st.info("No Excel files available for download.")
            
            # CSV files
            with download_tabs[1]:
                if ".csv" in grouped_files:
                    cols = st.columns(3)
                    for i, (file_name, file_path) in enumerate(sorted(grouped_files[".csv"])):
                        try:
                            with open(file_path, "rb") as file:
                                file_data = file.read()
                                    
                            cols[i % 3].download_button(
                                label=file_name,
                                data=file_data,
                                file_name=file_name,
                                mime=file_types[".csv"][1],
                                key=f"csv_{i}",
                                use_container_width=True
                            )
                        except Exception as e:
                            cols[i % 3].error(f"Error with file {file_name}: {str(e)}")
                else:
                    st.info("No CSV files available for download.")
            
            # Image files
            with download_tabs[2]:
                if ".png" in grouped_files:
                    cols = st.columns(3)
                    for i, (file_name, file_path) in enumerate(sorted(grouped_files[".png"])):
                        try:
                            with open(file_path, "rb") as file:
                                file_data = file.read()
                                # Show a thumbnail
                                cols[i % 3].image(file_data, caption=file_name, width=200)
                                # Add download button
                                cols[i % 3].download_button(
                                    label=f"Download {file_name}",
                                    data=file_data,
                                    file_name=file_name,
                                    mime=file_types[".png"][1],
                                    key=f"png_{i}",
                                    use_container_width=True
                                )
                        except Exception as e:
                            cols[i % 3].error(f"Error with file {file_name}: {str(e)}")
                else:
                    st.info("No image files available for download.")
    else:
        st.info("No files available for download.")

# Add help information to sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This is the Camp Northland Hobby Allocator tool. "
    "It helps automatically assign campers to their preferred hobby activities "
    "while optimizing overall satisfaction and meeting constraints."
)

# Add system status check to sidebar
with st.sidebar.expander("System Status", expanded=False):
    check_pulp_installation()
    st.text(f"App Version: {APP_VERSION}")
    st.text(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

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