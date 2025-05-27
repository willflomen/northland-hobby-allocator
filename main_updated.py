import os
import sys
import argparse
from datetime import datetime

# Add local bin to PATH for Streamlit Cloud
if '/home/appuser/.local/bin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/home/appuser/.local/bin:' + os.environ.get('PATH', '')

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("✓ Data processing libraries imported successfully")
except ImportError as e:
    print(f"✗ Failed to import data processing libraries: {e}")
    sys.exit(1)

try:
    import pulp
    print(f"PuLP imported successfully, version: {pulp.__version__}")
except ImportError as e:
    print(f"Failed to import PuLP: {e}")
    sys.exit(1)

def get_best_solver():
    """Get the best available solver for the current environment"""
    
    # Test solvers in order of preference
    solvers_to_try = [
        ('COIN_CMD', pulp.COIN_CMD),
        ('CBC_CMD', pulp.PULP_CBC_CMD),
        ('GLPK_CMD', pulp.GLPK_CMD),
    ]
    
    print("Testing available solvers...")
    
    for solver_name, solver_class in solvers_to_try:
        try:
            solver = solver_class(msg=False)
            if solver.available():
                print(f"✓ {solver_name} is available")
                return solver
            else:
                print(f"✗ {solver_name} is not available")
        except Exception as e:
            print(f"✗ {solver_name} failed with error: {e}")
    
    # Fallback to default solver
    print("⚠ Using default solver as fallback")
    return None

# Test solver availability at import time
BEST_SOLVER = get_best_solver()

class HobbyAllocator:
    def __init__(self):
        """Initialize the HobbyAllocator class"""
        self.campers_df = None
        self.hobby_config = None
        self.allocations = None
        self.satisfaction_score = None
        self.model = None
        self.choice_distribution = None
        self.pre_assignments = None
        self.previous_allocations = None
        self.weights = None
        self.restricted_hobbies = []
        
    def read_input_data(self, file_path):
        """
        Read camper preference data from CSV file
        
        Parameters:
        file_path (str): Path to the CSV file
        
        Returns:
        DataFrame: The loaded camper data
        """
        print(f"Reading camper data from {file_path}...")
        
        # Read the CSV file
        self.campers_df = pd.read_csv(file_path)
        
        # Print data overview
        print(f"Loaded {len(self.campers_df)} campers")
        print(f"Columns: {self.campers_df.columns.tolist()}")
        
        # Check for missing choice data
        choice_columns = [col for col in self.campers_df.columns if col.startswith('Choice')]
        missing_choices = self.campers_df[choice_columns].isna().sum().sum()
        total_choices = len(self.campers_df) * len(choice_columns)
        print(f"Missing choices: {missing_choices} out of {total_choices} ({missing_choices/total_choices*100:.1f}%)")
        
        return self.campers_df

    def read_hobby_config(self, file_path):
        """
        Read hobby configuration data
        
        Parameters:
        file_path (str): Path to the hobby config CSV file
        
        Returns:
        DataFrame: Hobby configuration data
        """
        print(f"Reading hobby configuration from {file_path}...")
        
        self.hobby_config = pd.read_csv(file_path)
        
        # Print config overview
        print(f"Loaded {len(self.hobby_config)} hobbies")
        print(f"Columns: {self.hobby_config.columns.tolist()}")
        
        # Handle various column naming conventions
        if 'Hobby Name' in self.hobby_config.columns and 'Name' not in self.hobby_config.columns:
            self.hobby_config['Name'] = self.hobby_config['Hobby Name']
            
        if 'Allowed Groups' in self.hobby_config.columns and 'Allowed Divisions' not in self.hobby_config.columns:
            self.hobby_config['Allowed Divisions'] = self.hobby_config['Allowed Groups']
        
        # Extract premium activities
        self.premium_activities = []
        
        # Check for Premium column
        if 'Premium' in self.hobby_config.columns:
            # Convert to string and lowercase for case-insensitive comparison
            premium_values = self.hobby_config['Premium'].astype(str).str.lower()
            premium_hobbies = self.hobby_config[premium_values.isin(['yes', 'true', '1', 'y'])]
            self.premium_activities = premium_hobbies['Name'].tolist()
            
            if self.premium_activities:
                print(f"Found premium activities in config: {self.premium_activities}")
        
        return self.hobby_config

    def read_pre_assignments(self, file_path):
        if file_path is None or not os.path.exists(file_path):
            print("No pre-assignments file provided or file does not exist.")
            self.pre_assignments = {}
            self.restricted_hobbies = []
            return self.pre_assignments
            
        print(f"Reading pre-assignments from {file_path}...")
        
        # Read the CSV file
        pre_assignments_df = pd.read_csv(file_path)
        
        # Print data overview
        print(f"Loaded {len(pre_assignments_df)} pre-assignments from file")
        
        # Create dictionary to map camper names to assigned hobbies
        name_to_hobby = {}
        for _, row in pre_assignments_df.iterrows():
            name_to_hobby[row['Full Name']] = row['Assigned Hobby']
        
        # Map to camper indices in the main dataframe
        self.pre_assignments = {}
        restricted_hobbies = set()
        not_found_campers = []
        
        # Check if a restricted_hobbies list was already provided (e.g., via command line)
        # If so, use that instead of checking the config file
        if hasattr(self, 'restricted_hobbies') and self.restricted_hobbies:
            print(f"Using pre-set restricted hobbies list: {self.restricted_hobbies}")
            # Convert existing list to a set for easier checking
            restricted_hobbies = set(self.restricted_hobbies)
        else:
            # Initialize an empty list if not already set
            self.restricted_hobbies = []
        
        for name, hobby in name_to_hobby.items():
            # Find matching camper in main dataframe
            matching_indices = self.campers_df.index[self.campers_df['Full Name'] == name].tolist()
            if matching_indices:
                camper_idx = matching_indices[0]
                self.pre_assignments[camper_idx] = hobby
                
                # Only check if this hobby is restricted if we don't have a pre-set list
                if not hasattr(self, 'restricted_hobbies') or not self.restricted_hobbies:
                    # Check if this hobby is marked as restricted in the config
                    hobby_row = self.hobby_config[self.hobby_config['Name'] == hobby]
                    if not hobby_row.empty:
                        # Check for Restricted column - default to Yes if column doesn't exist
                        is_restricted = True
                        if 'Restricted' in hobby_row.columns:
                            restricted_value = str(hobby_row['Restricted'].iloc[0]).lower()
                            is_restricted = restricted_value in ['yes', 'true', '1', 'y']
                        
                        if is_restricted:
                            restricted_hobbies.add(hobby)
            else:
                not_found_campers.append(name)
        
        if not_found_campers:
            print(f"WARNING: Could not find these campers in the main data file: {not_found_campers}")
            print("Check for exact name spelling (including spaces, capitalization, etc.)")
        
        # Update restricted_hobbies if we didn't have a pre-set list
        if not hasattr(self, 'restricted_hobbies') or not self.restricted_hobbies:
            self.restricted_hobbies = list(restricted_hobbies)
        
        print(f"Successfully processed {len(self.pre_assignments)} pre-assignments for campers")
        print(f"Restricted hobbies: {self.restricted_hobbies}")
        
        # Count pre-assignments per hobby
        hobby_pre_assignment_counts = {}
        for camper_idx, hobby in self.pre_assignments.items():
            if hobby not in hobby_pre_assignment_counts:
                hobby_pre_assignment_counts[hobby] = 0
            hobby_pre_assignment_counts[hobby] += 1
        
        # Store pre-assignment counts
        self.hobby_pre_assignment_counts = hobby_pre_assignment_counts
        
        if hobby_pre_assignment_counts:
            print("Pre-assignment counts by hobby:")
            for hobby, count in hobby_pre_assignment_counts.items():
                restricted_status = "restricted" if hobby in self.restricted_hobbies else "unrestricted"
                print(f"  - {hobby}: {count} campers ({restricted_status})")
        
        return self.pre_assignments
    
    def read_previous_allocations(self, file_path, weight_factor=0.2):
        """
        Read previous allocation data to prioritize campers who didn't get first choice
        
        Parameters:
        file_path (str): Path to the previous allocations CSV file
        weight_factor (float): Factor to increase weights for campers based on previous rank
        
        Returns:
        dict: Dictionary of priority weights for each camper
        """
        # Initialize default weights for all campers
        self.weights = {i: 1.0 for i in range(len(self.campers_df))}
        self.previous_allocations = {}
        
        if file_path is None or not os.path.exists(file_path):
            print("No previous allocations file provided or file does not exist.")
            return self.weights
            
        print(f"Reading previous allocations from {file_path}...")
        
        # Read the CSV file
        prev_allocations_df = pd.read_csv(file_path)
        
        # Print data overview
        print(f"Loaded {len(prev_allocations_df)} previous allocations")
        
        # Create dictionary to map camper names to previous choice ranks
        name_to_rank = {}
        for _, row in prev_allocations_df.iterrows():
            if 'Full Name' in row and 'Choice Rank' in row and pd.notna(row['Choice Rank']):
                try:
                    # Convert to int if possible
                    choice_rank = int(row['Choice Rank'])
                    name_to_rank[row['Full Name']] = choice_rank
                except (ValueError, TypeError):
                    # Skip if value can't be converted to int
                    print(f"Warning: Invalid Choice Rank value for {row['Full Name']}")
        
        # Set weights based on previous allocation ranks
        priority_campers_count = 0
        for i, row in self.campers_df.iterrows():
            if 'Full Name' in row and row['Full Name'] in name_to_rank:
                prev_rank = name_to_rank[row['Full Name']]
                self.previous_allocations[i] = prev_rank
                
                # Higher weights for campers who didn't get their top choices
                if prev_rank > 1:  # They didn't get first choice
                    self.weights[i] = 1.0 + weight_factor * (prev_rank - 1)
                    priority_campers_count += 1
                else:
                    self.weights[i] = 1.0  # Default weight
        
        # Calculate statistics on weights
        weights_list = list(self.weights.values())
        if weights_list:
            print(f"Weight range: {min(weights_list)} to {max(weights_list)}")
            print(f"Priority campers (with increased weights): {priority_campers_count}")
        
        return self.weights
    
    def create_allocation_model(self, premium_activities=None, premium_factor=0.1):
        """
        Create and solve the optimization model for hobby allocation
        
        Parameters:
        premium_activities (list): List of premium activities that should primarily go to first-choice campers
        premium_factor (float): Factor to reduce preference scores for non-first-choice premium activities
        
        Returns:
        tuple: (allocation dictionary, satisfaction score, model)
        """
        print("Creating optimization model...")
        
        if self.campers_df is None or self.hobby_config is None:
            raise ValueError("Camper data or hobby configuration not loaded")
        
        # Get list of campers and hobbies
        campers = list(range(len(self.campers_df)))
        hobbies = self.hobby_config['Name'].tolist()
        
        # Default premium activities list if not provided
        if premium_activities is None:
            premium_activities = []
        
        if premium_activities:
            print(f"Premium activities with special allocation: {premium_activities}")
        
        # Get divisions
        divisions = self.campers_df['Division'].tolist()
        unique_divisions = list(set(divisions))
        
        print(f"Processing {len(campers)} campers across {len(unique_divisions)} divisions")
        print(f"Allocating to {len(hobbies)} hobbies")
        
        # Create a preference matrix (5=first choice, 4=second choice, etc.)
        preferences = {}
        for i in campers:
            preferences[i] = {}
            row = self.campers_df.iloc[i]
            
            # Initialize all hobbies with preference 0
            for hobby in hobbies:
                preferences[i][hobby] = 0
            
            # Set preferences based on choices (handling missing values)
            for j in range(1, 6):
                choice_col = f'Choice {j}'
                if choice_col in row and pd.notna(row[choice_col]) and row[choice_col] in hobbies:
                    hobby = row[choice_col]
                    score = 6 - j  # 5 for first choice, 4 for second, etc.
                    
                    # For premium activities, drastically reduce value of non-first-choice preferences
                    if hobby in premium_activities and j > 1:  # Not first choice
                        score = score * premium_factor
                    
                    preferences[i][hobby] = score
        
        # Create the optimization model
        model = pulp.LpProblem("HobbyAllocation", pulp.LpMaximize)
        
        # Decision variables: 1 if camper i is assigned to hobby j, 0 otherwise
        x = {}
        for i in campers:
            for j in hobbies:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
        
        # Binary variables to track if at least 2 campers from a division are in a hobby
        y = {}
        for div in unique_divisions:
            for j in hobbies:
                y[(div, j)] = pulp.LpVariable(f"y_{div}_{j}", cat=pulp.LpBinary)
        
        # Initialize weights if not already set
        if not hasattr(self, 'weights') or self.weights is None:
            self.weights = {i: 1.0 for i in campers}
        
        # Objective: Maximize weighted preference satisfaction
        model += pulp.lpSum(self.weights[i] * preferences[i][j] * x[(i, j)] for i in campers for j in hobbies)
        
        # Constraint: Each camper is assigned to exactly one hobby
        for i in campers:
            model += pulp.lpSum(x[(i, j)] for j in hobbies) == 1
        
        # Special handling for pre-assigned campers - must happen first to override other constraints
        if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and self.pre_assignments:
            for i, hobby in self.pre_assignments.items():
                model += x[(i, hobby)] == 1
            
            # Handle restricted hobbies - prevent non-pre-assigned campers from being assigned to them
            if hasattr(self, 'restricted_hobbies') and self.restricted_hobbies:
                for hobby in self.restricted_hobbies:
                    allowed_campers = [i for i, h in self.pre_assignments.items() if h == hobby]
                    for i in campers:
                        if i not in allowed_campers:
                            model += x[(i, hobby)] == 0
        
        # Constraint: Hobby capacity limits
        for j in hobbies:
            hobby_row = self.hobby_config[self.hobby_config['Name'] == j].iloc[0]
            min_capacity = hobby_row['Min Capacity']
            max_capacity = hobby_row['Max Capacity']
            
            # Adjust capacity for non-restricted hobbies with pre-assignments
            pre_assigned_count = 0
            if hasattr(self, 'hobby_pre_assignment_counts') and j in self.hobby_pre_assignment_counts:
                pre_assigned_count = self.hobby_pre_assignment_counts[j]
                
                # Only adjust capacity for non-restricted hobbies
                if j not in self.restricted_hobbies:
                    # Adjust min and max capacity for non-restricted hobbies
                    max_capacity -= pre_assigned_count
                    # Don't let min capacity go below zero or become greater than max
                    min_capacity = max(0, min(min_capacity - pre_assigned_count, max_capacity))
                    
                    print(f"Adjusted capacity for {j} (non-restricted): min={min_capacity}, max={max_capacity}")
            
            # For non-restricted hobbies, we only need to apply constraints to non-pre-assigned campers
            if j in self.restricted_hobbies:
                # For restricted hobbies, no adjustment needed because only pre-assigned campers can be assigned
                # Min capacity constraint
                model += pulp.lpSum(x[(i, j)] for i in campers) >= min_capacity
                
                # Max capacity constraint
                model += pulp.lpSum(x[(i, j)] for i in campers) <= max_capacity
            else:
                # For non-restricted hobbies, exclude pre-assigned campers from capacity constraints
                non_preassigned_campers = [i for i in campers if i not in self.pre_assignments or self.pre_assignments[i] != j]
                
                # Min capacity constraint (excluding pre-assigned)
                model += pulp.lpSum(x[(i, j)] for i in non_preassigned_campers) >= min_capacity
                
                # Max capacity constraint (excluding pre-assigned)
                model += pulp.lpSum(x[(i, j)] for i in non_preassigned_campers) <= max_capacity
        
        # Constraint: Allowed divisions per hobby
        for j in hobbies:
            hobby_row = self.hobby_config[self.hobby_config['Name'] == j].iloc[0]
            
            # Get allowed groups/divisions
            if 'Allowed Divisions' in hobby_row:
                allowed_groups_str = hobby_row['Allowed Divisions']
            else:
                allowed_groups_str = hobby_row['Allowed Groups']
                
            # Handle different formats of allowed groups
            allowed_groups = []
            if pd.notna(allowed_groups_str):
                if isinstance(allowed_groups_str, str):
                    allowed_groups = [group.strip() for group in allowed_groups_str.split(',')]
                else:
                    # Handle case when it's not a string
                    allowed_groups = [str(allowed_groups_str)]
            
            # If allowed_groups is empty, assume all groups are allowed
            if not allowed_groups:
                continue
                
            for i in campers:
                # Skip division constraints for pre-assigned campers
                if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments and self.pre_assignments[i] == j:
                    continue
                
                if divisions[i] not in allowed_groups:
                    model += x[(i, j)] == 0
        
        # Constraint: At least 2 campers from the same division in each hobby
        # First, track for each division and hobby whether there are at least 2 campers
        for div in unique_divisions:
            for j in hobbies:
                # Get campers from this division
                div_campers = [i for i in campers if divisions[i] == div]
                
                # If there are fewer than 2 campers in this division, skip this constraint
                if len(div_campers) < 2:
                    continue
                
                # If there's only one camper assigned, y must be 0
                model += pulp.lpSum(x[(i, j)] for i in div_campers) >= 2 * y[(div, j)]
                
                # If there are two or more campers, y can be 1
                M = len(div_campers)  # Big M constant
                model += pulp.lpSum(x[(i, j)] for i in div_campers) <= M * y[(div, j)] + 1
        
        # Ensure a camper is only assigned to a hobby if their division has at least 2 campers there
        # UNLESS they are pre-assigned to that hobby
        for i in campers:
            div = divisions[i]
            div_campers = [c for c in campers if divisions[c] == div]
            
            # If there's only one camper in this division, skip the constraint
            if len(div_campers) < 2:
                continue
            
            # Check if this camper is pre-assigned
            is_preassigned = False
            preassigned_hobby = None
            if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments:
                is_preassigned = True
                preassigned_hobby = self.pre_assignments[i]
            
            for j in hobbies:
                # Skip this constraint for pre-assigned campers
                if is_preassigned and preassigned_hobby == j:
                    continue
                
                # For non-preassigned campers or other hobbies, apply the constraint
                model += x[(i, j)] <= y[(div, j)]
        
        # Special handling for pre-assigned campers' division constraints
        # This allows a single pre-assigned camper to be assigned to a hobby
        # even if they're the only one from their division
        if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and self.pre_assignments:
            for i, hobby in self.pre_assignments.items():
                # Force this assignment regardless of division constraints
                model += x[(i, hobby)] == 1
        
        print("Solving optimization model (this may take a few minutes)...")
        
        # Use the best available solver
        if BEST_SOLVER is not None:
            print(f"Using solver: {type(BEST_SOLVER).__name__}")
            result = model.solve(BEST_SOLVER)
        else:
            print("Using default solver")
            result = model.solve()
        
        if model.status != 1:
            print(f"WARNING: Model solution status: {model.status}")
            print(f"Status meaning: {pulp.LpStatus[model.status]}")
            if model.status == -1:
                print("The model is infeasible - there may be conflicting constraints")
            elif model.status == -2:
                print("The model is unbounded")
            elif model.status == 0:
                print("The model status is undefined - solver may have failed")
            print(f"Try adjusting the min/max capacities or allowed groups.")
        
        # Extract solution
        allocations = {}
        for i in campers:
            for j in hobbies:
                if pulp.value(x[(i, j)]) == 1:
                    allocations[i] = j
        
        # Calculate satisfaction score (percentage of campers getting their top 3 choices)
        total_score = 0
        top_choices = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 0: 0}  # To count each choice rank
        
        # Store allocations by choice for each hobby
        hobby_choice_counts = {}
        for hobby in hobbies:
            hobby_choice_counts[hobby] = {
                1: 0,  # First choice
                2: 0,  # Second choice
                3: 0,  # Third choice
                4: 0,  # Fourth choice
                5: 0,  # Fifth choice
                0: 0,  # No choice match
                'pre': 0  # Pre-assigned
            }
        
        for i in campers:
            if i in allocations:
                hobby = allocations[i]
                
                # Check if this was a pre-assignment
                is_preassigned = False
                if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments:
                    is_preassigned = True
                    hobby_choice_counts[hobby]['pre'] += 1
                    top_choices[0] += 1  # Count in "no choice" for overall stats
                    continue
                
                # Find which choice this was
                choice_rank = 0
                for j in range(1, 6):
                    choice_col = f'Choice {j}'
                    if choice_col in self.campers_df.columns and pd.notna(self.campers_df.loc[i, choice_col]) and self.campers_df.loc[i, choice_col] == hobby:
                        choice_rank = j
                        break
                
                # Update counts
                pref = 6 - choice_rank if choice_rank > 0 else 0
                total_score += pref
                
                if choice_rank in top_choices:
                    top_choices[choice_rank] += 1
                    hobby_choice_counts[hobby][choice_rank] += 1
                else:
                    top_choices[0] += 1
                    hobby_choice_counts[hobby][0] += 1
        
        satisfaction_score = (top_choices[1] + top_choices[2] + top_choices[3]) / len(campers) * 100
        
        self.allocations = allocations
        self.model = model
        self.satisfaction_score = satisfaction_score
        self.choice_distribution = top_choices
        self.hobby_choice_counts = hobby_choice_counts
        
        print(f"Model solved with satisfaction score: {satisfaction_score:.2f}%")
        print(f"First choice: {top_choices[1]} campers ({top_choices[1]/len(campers)*100:.1f}%)")
        print(f"Second choice: {top_choices[2]} campers ({top_choices[2]/len(campers)*100:.1f}%)")
        print(f"Third choice: {top_choices[3]} campers ({top_choices[3]/len(campers)*100:.1f}%)")
        print(f"Fourth choice: {top_choices[4]} campers ({top_choices[4]/len(campers)*100:.1f}%)")
        print(f"Fifth choice: {top_choices[5]} campers ({top_choices[5]/len(campers)*100:.1f}%)")
        print(f"No choice matched: {top_choices[0]} campers ({top_choices[0]/len(campers)*100:.1f}%)")
        
        # Special status: Check how many campers with priority got their first choice
        if hasattr(self, 'previous_allocations') and self.previous_allocations:
            priority_campers = [i for i, prev_rank in self.previous_allocations.items() if prev_rank > 1]
            priority_first_choice = sum(1 for i in priority_campers if i in allocations and 
                                    (hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments) or
                                    any(self.campers_df.loc[i, f'Choice {j}'] == allocations[i] for j in [1] if f'Choice {j}' in self.campers_df.columns))
            
            if priority_campers:
                print(f"\nPriority campers (didn't get first choice previously): {len(priority_campers)}")
                print(f"Priority campers who got first choice this time: {priority_first_choice} ({priority_first_choice/len(priority_campers)*100:.1f}%)")
        
        return allocations, satisfaction_score, model
    
    def generate_hobby_excel(self, output_path):
        """
        Generate Excel file organized by hobby
        
        Parameters:
        output_path (str): Path to save the Excel file
        """
        if self.allocations is None:
            raise ValueError("Model has not been solved yet")
            
        print(f"Generating hobby-organized Excel file at {output_path}...")
        
        # Create a dictionary to hold DataFrames for each hobby
        hobby_dfs = {}
        
        # Prepare the data for each hobby
        for camper_idx, hobby in self.allocations.items():
            if hobby not in hobby_dfs:
                hobby_dfs[hobby] = []
            
            camper = self.campers_df.iloc[camper_idx]
            
            # Check if Cabin column exists
            cabin = "Unknown"
            if 'Cabin' in self.campers_df.columns:
                cabin = camper['Cabin'] if pd.notna(camper['Cabin']) else "Unknown"
            
            # Check if pre-assigned
            pre_assigned = False
            if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and camper_idx in self.pre_assignments:
                pre_assigned = True
            
            hobby_dfs[hobby].append({
                'Name': camper['Full Name'],
                'Division': camper['Division'],
                'Cabin': cabin,
                'Pre-assigned': 'Yes' if pre_assigned else 'No'
            })
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Create a sheet for each hobby
            for hobby, campers in hobby_dfs.items():
                if not campers:
                    continue
                    
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(campers)
                
                # Sort by Division and then Name
                if 'Division' in df.columns and 'Name' in df.columns:
                    df = df.sort_values(by=['Division', 'Name'])
                
                # Write to Excel, using hobby name as sheet name
                safe_sheet_name = str(hobby)[:31]  # Excel sheet names have a 31 character limit
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    
    def generate_division_excel(self, output_path):
        """
        Generate Excel file organized by division
        
        Parameters:
        output_path (str): Path to save the Excel file
        """
        if self.allocations is None:
            raise ValueError("Model has not been solved yet")
            
        print(f"Generating division-organized Excel file at {output_path}...")
        
        # Create a dictionary to hold DataFrames for each division
        division_dfs = {}
        
        # Prepare the data for each division
        for camper_idx, hobby in self.allocations.items():
            camper = self.campers_df.iloc[camper_idx]
            division = camper['Division']
            
            if division not in division_dfs:
                division_dfs[division] = []
            
            # Check if Cabin column exists
            cabin = "Unknown"
            if 'Cabin' in self.campers_df.columns:
                cabin = camper['Cabin'] if pd.notna(camper['Cabin']) else "Unknown"
            
            # Check if pre-assigned
            pre_assigned = False
            if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and camper_idx in self.pre_assignments:
                pre_assigned = True
            
            division_dfs[division].append({
                'Cabin': cabin,
                'Name': camper['Full Name'],
                'Assigned Hobby': hobby,
                'Pre-assigned': 'Yes' if pre_assigned else 'No'
            })
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Create a sheet for each division
            for division, campers in division_dfs.items():
                if not campers:
                    continue
                    
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(campers)
                
                # Sort by Cabin and then Name
                if 'Cabin' in df.columns and 'Name' in df.columns:
                    df = df.sort_values(by=['Cabin', 'Name'])
                
                # Write to Excel, using division name as sheet name
                safe_sheet_name = str(division)[:31]  # Excel sheet names have a 31 character limit
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    
    def generate_summary(self, output_path):
        """
        Generate summary file with hobby statistics
        
        Parameters:
        output_path (str): Path to save the summary file
        
        Returns:
        DataFrame: Summary data for visualization
        """
        if self.allocations is None:
            raise ValueError("Model has not been solved yet")
            
        print(f"Generating summary file at {output_path}...")
        
        # Count campers assigned to each hobby
        hobby_counts = {}
        pre_assigned_counts = {}
        
        for i, hobby in self.allocations.items():
            # Count total campers
            if hobby in hobby_counts:
                hobby_counts[hobby] += 1
            else:
                hobby_counts[hobby] = 1
                
            # Count pre-assigned campers
            if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments:
                if hobby in pre_assigned_counts:
                    pre_assigned_counts[hobby] += 1
                else:
                    pre_assigned_counts[hobby] = 1
        
        # Create summary DataFrame
        summary_data = []
        for _, row in self.hobby_config.iterrows():
            hobby_name = row['Name']
            assigned = hobby_counts.get(hobby_name, 0)
            pre_assigned = pre_assigned_counts.get(hobby_name, 0)
            
            # Handle different column names
            location = row.get('Location', 'Unknown')
            leader = row.get('Leader', row.get('Specialty', 'Unknown'))
            allowed_groups = row.get('Allowed Groups', row.get('Allowed Divisions', 'All'))
            
            min_capacity = row['Min Capacity']
            max_capacity = row['Max Capacity']
            
            # Check if hobby is restricted
            is_restricted = "No"
            if 'Restricted' in row:
                restricted_value = str(row['Restricted']).lower()
                is_restricted = "Yes" if restricted_value in ['yes', 'true', '1', 'y'] else "No"
            else:
                # If Restricted column doesn't exist, check if it's in the restricted_hobbies list
                is_restricted = "Yes" if hobby_name in self.restricted_hobbies else "No"
            
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
        summary_df.to_csv(output_path, index=False)
        
        return summary_df
    
    def create_allocation_bar_chart(self, summary_df, output_path):
        """
        Create a bar chart showing the allocation results
        
        Parameters:
        summary_df (DataFrame): Summary data with hobby statistics
        output_path (str): Path to save the chart image
        """
        print(f"Creating hobby allocation visualization at {output_path}...")
        
        # Extract data for plotting
        hobbies = summary_df['Hobby Name'].tolist()
        assigned = summary_df['Number of Campers Assigned'].tolist()
        pre_assigned = summary_df['Pre-assigned Campers'].tolist()
        regular_assigned = summary_df['Regular Assigned Campers'].tolist()
        min_capacities = summary_df['Min Capacity'].tolist()
        max_capacities = summary_df['Max Capacity'].tolist()
        
        # Get restricted status if available
        restricted_status = []
        if 'Restricted' in summary_df.columns:
            restricted_status = summary_df['Restricted'].tolist()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot the stacked bars
        bar_width = 0.6
        bar_positions = range(len(hobbies))
        
        # Create stacked bars for pre-assigned and regular campers
        bars1 = ax.bar(bar_positions, regular_assigned, bar_width, label='Regular Assigned Campers', color='#3498db')
        bars2 = ax.bar(bar_positions, pre_assigned, bar_width, bottom=regular_assigned, label='Pre-assigned Campers', color='#9b59b6')
        
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
        
        # Mark restricted hobbies with an asterisk if restricted_status is available
        if restricted_status:
            for i, is_restricted in enumerate(restricted_status):
                if is_restricted == "Yes":
                    ax.text(i, -2, "*", ha='center', va='top', fontsize=24, color='red')
        
        # Set axis labels and title
        ax.set_xlabel('Hobby', fontsize=12)
        ax.set_ylabel('Number of Campers', fontsize=12)
        ax.set_title('Camper Allocation by Hobby', fontsize=16)
        
        # Set x-axis tick labels to hobby names
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(hobbies, rotation=45, ha='right', fontsize=10)
        
        # Add legend
        legend_elements = [bars1, bars2, min_cap_line, max_cap_line]
        legend_labels = ['Regular Assigned Campers', 'Pre-assigned Campers', 'Min Capacity', 'Max Capacity']
        
        # Add note about restricted hobbies if applicable
        if any(status == "Yes" for status in restricted_status):
            ax.text(0.01, 0.01, "* Restricted hobbies (only pre-assigned campers)", transform=ax.transAxes, 
                   fontsize=10, color='red', ha='left')
        
        ax.legend(legend_elements, legend_labels)
        
        # Add grid for easier reading
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()

    def create_choice_distribution_chart(self, output_path):
        if self.choice_distribution is None:
            raise ValueError("Model has not been solved yet")
            
        print(f"Creating choice distribution visualization at {output_path}...")
        
        # Extract data for counting pre-assigned vs. regular allocations
        regular_choices = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 0: 0}  # Regular campers
        pre_assigned_count = 0  # Pre-assigned campers
        
        # Count each camper's choice rank, separating pre-assigned campers
        for i, hobby in self.allocations.items():
            # Check if this was a pre-assignment
            if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments:
                pre_assigned_count += 1
                continue  # Skip pre-assigned campers from regular counts
            
            # Calculate choice rank for regular campers
            choice_rank = 0
            for j in range(1, 6):
                choice_col = f'Choice {j}'
                if choice_col in self.campers_df.columns and pd.notna(self.campers_df.loc[i, choice_col]) and self.campers_df.loc[i, choice_col] == hobby:
                    choice_rank = j
                    break
                    
            # Increment the appropriate counter
            if choice_rank in regular_choices:
                regular_choices[choice_rank] += 1
        
        # Extract data for plotting
        choices = ['First Choice', 'Second Choice', 'Third Choice', 'Fourth Choice', 'Fifth Choice', 'No Choice Match', 'Pre-assigned']
        counts = [
            regular_choices[1],  # First choice
            regular_choices[2],  # Second choice
            regular_choices[3],  # Third choice
            regular_choices[4],  # Fourth choice
            regular_choices[5],  # Fifth choice
            regular_choices[0],  # No choice match
            pre_assigned_count   # Pre-assigned campers
        ]
        
        # Calculate percentages
        total_campers = sum(counts)
        percentages = [count / total_campers * 100 for count in counts]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Define colors for the bars
        colors = ['#2ecc71', '#27ae60', '#3498db', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        
        # Plot the bars
        bar_positions = range(len(choices))
        bars = ax.bar(bar_positions, counts, color=colors)
        
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
        
        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()

    def create_hobby_choice_distribution_chart(self, output_path):
        """
        Create a stacked bar chart showing the distribution of choice ranks for each hobby
        
        Parameters:
        output_path (str): Path to save the chart image
        """
        if not hasattr(self, 'hobby_choice_counts') or not self.hobby_choice_counts:
            raise ValueError("Model has not been solved or hobby choice counts not calculated")
            
        print(f"Creating hobby choice distribution visualization at {output_path}...")
        
        # Get hobbies and their data
        hobbies = list(self.hobby_choice_counts.keys())
        
        # Extract data for plotting
        data = {
            '1st Choice': [self.hobby_choice_counts[h][1] for h in hobbies],
            '2nd Choice': [self.hobby_choice_counts[h][2] for h in hobbies],
            '3rd Choice': [self.hobby_choice_counts[h][3] for h in hobbies],
            '4th Choice': [self.hobby_choice_counts[h][4] for h in hobbies],
            '5th Choice': [self.hobby_choice_counts[h][5] for h in hobbies],
            'No Choice': [self.hobby_choice_counts[h][0] for h in hobbies],
            'Pre-assigned': [self.hobby_choice_counts[h]['pre'] for h in hobbies]
        }
        
        # Get total counts for each hobby for sorting
        total_counts = [sum(self.hobby_choice_counts[h].values()) for h in hobbies]
        
        # Sort hobbies by total campers (descending)
        sorted_indices = np.argsort(total_counts)[::-1]
        hobbies = [hobbies[i] for i in sorted_indices]
        
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
            bottom += data[category]
        
        # Add total count labels on top of stacked bars
        for i, total in enumerate(total_counts):
            if total > 0:  # Only label bars with campers
                ax.text(i, total + 1, str(total), ha='center', va='bottom', fontweight='bold')
        
        # Add data values to each segment of the stacked bars
        for category in stacking_order:
            for i, value in enumerate(data[category]):
                if value > 0:  # Only label segments with campers
                    # Calculate the middle position of the segment for text placement
                    pos = bottom[i] - data[category][i]/2
                    # Add text with smaller font if the segment is narrow
                    if data[category][i] > 2:  # Only add text if segment is tall enough
                        ax.text(i, pos, str(value), ha='center', va='center', fontweight='bold', 
                            color='white' if category in ['Pre-assigned'] else 'black',
                            fontsize=8 if data[category][i] < 5 else 10)
        
        # Mark restricted hobbies if available
        if hasattr(self, 'restricted_hobbies') and self.restricted_hobbies:
            restricted_indices = [i for i, h in enumerate(hobbies) if h in self.restricted_hobbies]
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
        
        # Save the figure
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def generate_master_excel(self, output_path):
        """
        Generate master Excel file with original data plus assignment details
        
        Parameters:
        output_path (str): Path to save the Excel file
        """
        if self.allocations is None:
            raise ValueError("Model has not been solved yet")
            
        print(f"Generating master Excel file at {output_path}...")
        
        # Create a copy of the original DataFrame
        master_df = self.campers_df.copy()
        
        # Add new columns for assigned hobby and choice rank
        master_df['Assigned Hobby'] = None
        master_df['Choice Rank'] = None
        master_df['Pre-assigned'] = 'No'
        master_df['Priority Weight'] = 1.0
        master_df['Previous Choice Rank'] = None
        
        # Fill in the assigned hobby and choice rank for each camper
        for i, hobby in self.allocations.items():
            master_df.loc[i, 'Assigned Hobby'] = hobby
            
            # Check if this was a pre-assignment
            if hasattr(self, 'pre_assignments') and self.pre_assignments is not None and i in self.pre_assignments:
                master_df.loc[i, 'Pre-assigned'] = 'Yes'
            
            # Fill in priority weight if applicable
            if hasattr(self, 'weights') and self.weights is not None and i in self.weights:
                master_df.loc[i, 'Priority Weight'] = self.weights[i]
            
            # Fill in previous choice rank if applicable
            if hasattr(self, 'previous_allocations') and self.previous_allocations is not None and i in self.previous_allocations:
                master_df.loc[i, 'Previous Choice Rank'] = self.previous_allocations[i]
            
            # Determine which choice rank this was
            choice_rank = 0  # Default to 0 if not found in choices
            for j in range(1, 6):
                choice_col = f'Choice {j}'
                if choice_col in master_df.columns and master_df.loc[i, choice_col] == hobby:
                    choice_rank = j
                    break
            
            master_df.loc[i, 'Choice Rank'] = choice_rank
        
        # Save to Excel
        master_df.to_excel(output_path, index=False)
        
        return master_df
    
    def generate_next_week_allocations(self, output_path):
        """
        Generate a file that can be used as previous allocations for next week
        
        Parameters:
        output_path (str): Path to save the file
        """
        if self.allocations is None:
            raise ValueError("Model has not been solved yet")
            
        print(f"Generating next week's previous allocations file at {output_path}...")
        
        # Create a DataFrame with just the necessary columns
        next_week_df = pd.DataFrame()
        next_week_df['Full Name'] = self.campers_df['Full Name']
        next_week_df['Division'] = self.campers_df['Division']
        next_week_df['Assigned Hobby'] = None
        next_week_df['Choice Rank'] = None
        
        # Fill in the assigned hobby and choice rank for each camper
        for i, hobby in self.allocations.items():
            next_week_df.loc[i, 'Assigned Hobby'] = hobby
            
            # Determine which choice rank this was
            choice_rank = 0  # Default to 0 if not found in choices
            for j in range(1, 6):
                choice_col = f'Choice {j}'
                if choice_col in self.campers_df.columns and self.campers_df.loc[i, choice_col] == hobby:
                    choice_rank = j
                    break
            
            next_week_df.loc[i, 'Choice Rank'] = choice_rank
        
        # Save to CSV
        next_week_df.to_csv(output_path, index=False)
        
        return next_week_df
    
    def generate_priority_comparison(self, output_path):
        """
        Generate a comparison file showing how priorities were applied
        
        Parameters:
        output_path (str): Path to save the Excel file
        """
        if self.allocations is None:
            raise ValueError("Model has not been solved yet")

        if not hasattr(self, 'previous_allocations') or not self.previous_allocations:
            print("No previous allocations data available for priority comparison.")
            # Create an empty DataFrame with a message
            comparison_df = pd.DataFrame({
                'Message': ['No previous allocations data available. Run with --previous-allocations parameter for priority tracking.']
            })
            comparison_df.to_excel(output_path, index=False)
            return comparison_df
            
        print(f"Generating priority comparison at {output_path}...")
        
        # Create DataFrame to track priorities and results
        comparison_data = []
        
        # Find campers who didn't get first choice previously (priority campers)
        priority_campers = [i for i, prev_rank in self.previous_allocations.items() if prev_rank > 1]
        
        for i in priority_campers:
            if i not in self.allocations:
                continue  # Skip if camper wasn't allocated for some reason
                
            # Get current allocation data
            camper = self.campers_df.loc[i]
            name = camper['Full Name'] if 'Full Name' in camper else f"Camper {i}"
            division = camper['Division'] if 'Division' in camper else "Unknown"
            
            # Get previous choice rank
            prev_rank = self.previous_allocations[i]
            
            # Get current choice rank
            current_hobby = self.allocations[i]
            current_rank = 0
            for j in range(1, 6):
                choice_col = f'Choice {j}'
                if choice_col in self.campers_df.columns and pd.notna(camper[choice_col]) and camper[choice_col] == current_hobby:
                    current_rank = j
                    break
            
            # Get priority weight
            weight = self.weights[i] if i in self.weights else 1.0
            
            # Pre-assigned status
            is_preassigned = "Yes" if (hasattr(self, 'pre_assignments') and 
                                    self.pre_assignments is not None and 
                                    i in self.pre_assignments) else "No"
            
            # Add to comparison data
            comparison_data.append({
                'Name': name,
                'Division': division,
                'Previous Choice Rank': prev_rank,
                'Current Choice Rank': current_rank,
                'Priority Weight': weight,
                'Pre-assigned': is_preassigned,
                'Improved': "Yes" if current_rank < prev_rank else 
                        "Same" if current_rank == prev_rank else "No",
                'Got First Choice': "Yes" if current_rank == 1 else "No"
            })
        
        # Create DataFrame and sort it
        comparison_df = pd.DataFrame(comparison_data)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(by=['Division', 'Name'])
        
        # Add summary statistics
        if not comparison_df.empty:
            improved_count = sum(1 for val in comparison_df['Improved'] if val == "Yes") 
            same_count = sum(1 for val in comparison_df['Improved'] if val == "Same")
            worse_count = sum(1 for val in comparison_df['Improved'] if val == "No")
            first_choice_count = sum(1 for val in comparison_df['Got First Choice'] if val == "Yes")
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Total Priority Campers', 
                    'Improved', 
                    'Same', 
                    'Worse',
                    'Got First Choice'
                ],
                'Count': [
                    len(comparison_df),
                    improved_count,
                    same_count,
                    worse_count,
                    first_choice_count
                ],
                'Percentage': [
                    100.0,
                    improved_count / len(comparison_df) * 100 if len(comparison_df) > 0 else 0,
                    same_count / len(comparison_df) * 100 if len(comparison_df) > 0 else 0,
                    worse_count / len(comparison_df) * 100 if len(comparison_df) > 0 else 0,
                    first_choice_count / len(comparison_df) * 100 if len(comparison_df) > 0 else 0
                ]
            })
            
            # Write to Excel with two sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                comparison_df.to_excel(writer, sheet_name='Priority Campers', index=False)
                stats_df.to_excel(writer, sheet_name='Summary', index=False)
                
            # Print summary to console
            print(f"\nPriority Comparison Summary:")
            print(f"  Total priority campers: {len(comparison_df)}")
            if len(comparison_df) > 0:
                print(f"  Improved: {improved_count} ({improved_count/len(comparison_df)*100:.1f}%)")
                print(f"  Same: {same_count} ({same_count/len(comparison_df)*100:.1f}%)")
                print(f"  Worse: {worse_count} ({worse_count/len(comparison_df)*100:.1f}%)")
                print(f"  Got first choice: {first_choice_count} ({first_choice_count/len(comparison_df)*100:.1f}%)")
        else:
            # No priority campers found
            comparison_df = pd.DataFrame({
                'Message': ['No priority campers found. All campers either got their first choice previously or are new.']
            })
            comparison_df.to_excel(output_path, index=False)
            print("No priority campers found in the data.")
        
        return comparison_df

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Camp Northland Hobby Allocator')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with camper preferences')
    parser.add_argument('--config', type=str, required=True, help='Path to hobby configuration CSV file')
    parser.add_argument('--pre-assignments', type=str, help='Path to pre-assignments CSV file')
    parser.add_argument('--previous-allocations', type=str, help='Path to previous allocations CSV file')
    parser.add_argument('--weight-factor', type=float, default=0.2, help='Weight factor for prioritizing campers')
    parser.add_argument('--premium-activities', type=str, help='Comma-separated list of premium activities (overrides config file)')
    parser.add_argument('--restricted-activities', type=str, help='Comma-separated list of restricted activities (overrides config file)')
    parser.add_argument('--premium-factor', type=float, default=0.1, help='Factor for non-first choice premium activities')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for output files')
    args = parser.parse_args()
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize allocator
    allocator = HobbyAllocator()
    
    # Read input data
    allocator.read_input_data(args.input)
    allocator.read_hobby_config(args.config)
    
    # Read optional data if provided
    if args.pre_assignments:
        allocator.read_pre_assignments(args.pre_assignments)
        
    if args.previous_allocations:
        allocator.read_previous_allocations(args.previous_allocations, args.weight_factor)
    
    # Get premium activities
    # First check if defined in config file
    premium_activities = getattr(allocator, 'premium_activities', None)
    premium_factor = args.premium_factor
    
    # Then override with command line if provided
    if args.premium_activities:
        premium_activities = [activity.strip() for activity in args.premium_activities.split(',')]
        print(f"Using premium activities from command line: {premium_activities}")
    
    # Handle restricted activities (similar to premium activities)
    if args.restricted_activities:
        restricted_activities = [activity.strip() for activity in args.restricted_activities.split(',')]
        print(f"Using restricted activities from command line: {restricted_activities}")
        
        # Update the restricted_hobbies list in the allocator
        allocator.restricted_hobbies = restricted_activities
        print(f"Updated restricted hobbies: {allocator.restricted_hobbies}")
    
    # Create and solve the model
    allocator.create_allocation_model(premium_activities=premium_activities, premium_factor=premium_factor)
    
    # Generate output files
    hobby_excel_path = os.path.join(output_dir, 'hobby_allocation.xlsx')
    allocator.generate_hobby_excel(hobby_excel_path)
    
    division_excel_path = os.path.join(output_dir, 'division_allocation.xlsx')
    allocator.generate_division_excel(division_excel_path)
    
    summary_path = os.path.join(output_dir, 'allocation_summary.csv')
    summary_df = allocator.generate_summary(summary_path)
    
    hobby_viz_path = os.path.join(output_dir, 'allocation_chart.png')
    allocator.create_allocation_bar_chart(summary_df, hobby_viz_path)
    
    choice_viz_path = os.path.join(output_dir, 'choice_distribution.png')
    allocator.create_choice_distribution_chart(choice_viz_path)
    
    hobby_choice_viz_path = os.path.join(output_dir, 'hobby_choice_distribution.png')
    allocator.create_hobby_choice_distribution_chart(hobby_choice_viz_path)
    
    master_path = os.path.join(output_dir, 'master_allocation.xlsx')
    allocator.generate_master_excel(master_path)
    
    next_week_path = os.path.join(output_dir, 'next_week_previous_allocations.csv')
    allocator.generate_next_week_allocations(next_week_path)
    
    # Generate priority comparison if applicable
    priority_path = os.path.join(output_dir, 'priority_comparison.xlsx')
    allocator.generate_priority_comparison(priority_path)
    
    print("\nAllocation completed successfully!")
    print(f"Satisfaction score: {allocator.satisfaction_score:.2f}%")
    
    # Print premium activities information if applicable
    if premium_activities:
        print(f"\nPremium activities: {', '.join(premium_activities)}")
        print(f"Premium factor: {premium_factor} (lower values make premium activities more exclusive to first choice)")
    
    # Print restricted hobbies information if applicable
    if hasattr(allocator, 'restricted_hobbies') and allocator.restricted_hobbies:
        print(f"\nRestricted hobbies: {', '.join(allocator.restricted_hobbies)}")
        print(f"These hobbies are limited to pre-assigned campers only")
    
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - Summary file: allocation_summary.csv")
    print(f"  - Hobby allocation: hobby_allocation.xlsx")
    print(f"  - Division allocation: division_allocation.xlsx")
    print(f"  - Hobby chart: allocation_chart.png")
    print(f"  - Choice distribution chart: choice_distribution.png")
    print(f"  - Hobby choice distribution chart: hobby_choice_distribution.png")
    print(f"  - Master allocation file: master_allocation.xlsx")
    print(f"  - Next week's previous allocations: next_week_previous_allocations.csv")
    print(f"  - Priority comparison: priority_comparison.xlsx")

if __name__ == "__main__":
    main()