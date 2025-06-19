import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import seaborn
import streamlit as st


# Load data
@st.cache_data
def load_data():
    excel_data = pd.ExcelFile("C:\All Working Folders\Autoroastering\OrionCare Data.xlsx")
    employee_df = excel_data.parse('Tb_EmployeeDetails')
    service_df = excel_data.parse('Tb_ServiceMaster')
    return employee_df, service_df

employee_df, service_df = load_data()

# Prepare employee features
employee_features = employee_df[[
    'EmployeeDetailId', 'EmployeeCode', 'FirstName', 'LastName',
    'Gender', 'PermanentStateId', 'PermanentPostcode']].copy()
employee_features['Availability'] = 1
employee_features['Reliability'] = np.random.uniform(0.6, 1.0, len(employee_features))

# Parameters
NUM_DAYS = 7
NUM_SHIFTS = 3
NUM_SERVICES = 3
SHIFT_LABELS = {0: 'Morning', 1: 'Evening', 2: 'Night'}

sample_services = service_df.head(NUM_SERVICES)[['ServiceId', 'ServiceName']].reset_index(drop=True)
employee_ids = employee_features['EmployeeDetailId'].tolist()
service_ids = sample_services['ServiceId'].tolist()

# Environment Class
class RosterEnv:
    def __init__(self, num_days=NUM_DAYS, num_shifts=NUM_SHIFTS):
        self.num_days = num_days
        self.num_shifts = num_shifts
        self.current_day = 0
        self.current_shift = 0
        self.roster = {}

    def reset(self):
        self.current_day = 0
        self.current_shift = 0
        self.roster = {}
        return self.get_state()

    def get_state(self):
        return (self.current_day, self.current_shift)

    def get_available_actions(self):
        return [(emp_id, svc_id) for emp_id in employee_ids for svc_id in service_ids]

    def step(self, action):
        emp_id, svc_id = action
        reliability = employee_features[employee_features['EmployeeDetailId'] == emp_id]['Reliability'].values[0]
        reward = 10 * reliability
        key = (self.current_day, self.current_shift)

        if key in self.roster and emp_id in [entry[0] for entry in self.roster[key]]:
            reward = -10

        if key not in self.roster:
            self.roster[key] = []
        self.roster[key].append((emp_id, svc_id))

        self.current_shift += 1
        if self.current_shift >= self.num_shifts:
            self.current_shift = 0
            self.current_day += 1

        done = self.current_day >= self.num_days
        return self.get_state(), reward, done

# Q-Learning Algorithm
def train_roster_agent(episodes=2000):
    Q = defaultdict(float)
    env = RosterEnv()
    alpha, gamma, epsilon = 0.1, 0.95, 0.1

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            available_actions = env.get_available_actions()
            if random.uniform(0, 1) < epsilon:
                action = random.choice(available_actions)
            else:
                q_vals = [Q[(state, a)] for a in available_actions]
                max_q = max(q_vals)
                best_actions = [a for a, q in zip(available_actions, q_vals) if q == max_q]
                action = random.choice(best_actions)

            next_state, reward, done = env.step(action)
            future_qs = [Q[(next_state, a)] for a in available_actions]
            max_future_q = max(future_qs) if future_qs else 0
            Q[(state, action)] += alpha * (reward + gamma * max_future_q - Q[(state, action)])
            state = next_state

    return env.roster

# Generate Roster and Create DataFrame
roster = train_roster_agent()
records = []
for (day, shift), assignments in roster.items():
    for emp_id, svc_id in assignments:
        emp_name = employee_features.loc[employee_features['EmployeeDetailId'] == emp_id, 'FirstName'].values[0]
        svc_name = sample_services.loc[sample_services['ServiceId'] == svc_id, 'ServiceName'].values[0]
        reliability = employee_features.loc[employee_features['EmployeeDetailId'] == emp_id, 'Reliability'].values[0]
        records.append({
            'Day': day + 1,
            'Shift': SHIFT_LABELS.get(shift, f'Shift {shift}'),
            'Employee': emp_name,
            'Reliability': reliability,
            'Service': svc_name
        })

roster_df = pd.DataFrame(records)

# Streamlit UI
st.title("Auto-Rostering with Reinforcement Learning")
st.write("### Final 7-Day, 3-Shift Roster")
st.dataframe(roster_df)

# Charts
st.write("### Total Shift Assignments per Employee")
assignment_counts = roster_df['Employee'].value_counts()
st.bar_chart(assignment_counts)

st.write("### Reliability Score Distribution")
fig, ax = plt.subplots()
sns.histplot(roster_df['Reliability'], kde=True, bins=10, ax=ax)
st.pyplot(fig)

st.write("### Service Distribution by Shift")
fig2, ax2 = plt.subplots()
sns.countplot(data=roster_df, x='Shift', hue='Service', ax=ax2)
plt.xticks(rotation=0)
st.pyplot(fig2)

#-------------------------------------------------------------#

# Upload day-shift configuration
st.sidebar.title("Upload Shift Requirements")
shift_file = st.sidebar.file_uploader("Upload Shift Requirement File (.csv)", type=["csv"])

if shift_file is not None:
    shift_config = pd.read_csv(shift_file, header=None)
    shift_config.columns = ['Morning Shift', 'Evening Shift', 'Night Shift'][:shift_config.shape[1]]
    shift_config.index = [f'Day {i+1}' for i in range(shift_config.shape[0])]

    with st.sidebar:
        st.write("### Uploaded Shift Matrix")
        st.dataframe(shift_config)
        st.markdown("**Note:** Numbers indicate how many staff are required for that shift (e.g., 2 = 2 staff)")

    NUM_DAYS, NUM_SHIFTS = shift_config.shape

    run_roster = st.button("Start Auto-Rostering", use_container_width=True)
    if not run_roster:
        st.stop()
else:
    st.warning("Please upload a shift requirement CSV file where 1 indicates shift required, and 0 means no shift.")
    st.stop()


# Load data
@st.cache_data
def load_data():
    excel_data = pd.ExcelFile("OrionCare Data.xlsx")
    employee_df = excel_data.parse('Tb_EmployeeDetails')
    service_df = excel_data.parse('Tb_ServiceMaster')
    return employee_df, service_df

employee_df, service_df = load_data()

# Prepare employee features
employee_features = employee_df[[
    'EmployeeDetailId', 'EmployeeCode', 'FirstName', 'LastName',
    'Gender', 'PermanentStateId', 'PermanentPostcode']].copy()
employee_features['Availability'] = 1
employee_features['Reliability'] = np.random.uniform(0.6, 1.0, len(employee_features))

NUM_SERVICES = 3
SHIFT_LABELS = {0: 'Morning', 1: 'Evening', 2: 'Night'}

sample_services = service_df.head(NUM_SERVICES)[['ServiceId', 'ServiceName']].reset_index(drop=True)
employee_ids = employee_features['EmployeeDetailId'].tolist()
service_ids = sample_services['ServiceId'].tolist()

# Environment Class
class RosterEnv:
    def __init__(self, shift_config, availability_lookup = None, enforce_no_consecutive=True):
        self.shift_config = shift_config.values
        self.num_days, self.num_shifts = self.shift_config.shape
        self.current_day = 0
        self.current_shift = 0
        self.roster = {}
        self.enforce_no_consecutive = enforce_no_consecutive

    def reset(self):
        self.current_day = 0
        self.current_shift = 0
        self.roster = {}
        return self.get_state()

    def get_state(self):
        return (self.current_day, self.current_shift)

    def get_available_actions(self):
        return [(emp_id, svc_id) for emp_id in employee_ids for svc_id in service_ids]

    def step(self, action):
        while self.current_day < self.num_days and self.shift_config[self.current_day][self.current_shift] == 0:
            self.current_shift += 1
            if self.current_shift >= self.num_shifts:
                self.current_shift = 0
                self.current_day += 1
            if self.current_day >= self.num_days:
                return self.get_state(), 0, True

        emp_id, svc_id = action
        reliability = employee_features[employee_features['EmployeeDetailId'] == emp_id]['Reliability'].values[0]
        reward = 10 * reliability
        key = (self.current_day, self.current_shift)

        if key not in self.roster:
            self.roster[key] = []

        already_assigned = emp_id in [entry[0] for entry in self.roster[key]]

        assigned_last_shift = False
        if self.enforce_no_consecutive:
            assigned_last_shift = (
                self.current_shift > 0 and (self.current_day, self.current_shift - 1) in self.roster and
                emp_id in [entry[0] for entry in self.roster[(self.current_day, self.current_shift - 1)]]
            ) or (
                self.current_shift == 0 and self.current_day > 0 and (self.current_day - 1, self.num_shifts - 1) in self.roster and
                emp_id in [entry[0] for entry in self.roster[(self.current_day - 1, self.num_shifts - 1)]]
            )

        if already_assigned or assigned_last_shift:
            reward = -10
        else:
            self.roster[key].append((emp_id, svc_id))

        if len(self.roster[key]) >= self.shift_config[self.current_day][self.current_shift]:
            self.current_shift += 1
            if self.current_shift >= self.num_shifts:
                self.current_shift = 0
                self.current_day += 1

        done = self.current_day >= self.num_days
        return self.get_state(), reward, done
    

def train_roster_agent(shift_config, episodes=1000):
    Q = defaultdict(float)
    enforce_no_consecutive = st.sidebar.checkbox("Disallow Consecutive Shifts", value=True)
    env = RosterEnv(shift_config, enforce_no_consecutive=enforce_no_consecutive)
    alpha, gamma, epsilon = 0.1, 0.95, 0.2

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            available_actions = env.get_available_actions()
            if random.uniform(0, 1) < epsilon:
                action = random.choice(available_actions)
            else:
                q_vals = [Q[(state, a)] for a in available_actions]
                max_q = max(q_vals)
                best_actions = [a for a, q in zip(available_actions, q_vals) if q == max_q]
                action = random.choice(best_actions)

            next_state, reward, done = env.step(action)
            future_qs = [Q[(next_state, a)] for a in available_actions]
            max_future_q = max(future_qs) if future_qs else 0
            Q[(state, action)] += alpha * (reward + gamma * max_future_q - Q[(state, action)])
            state = next_state

    return env.roster



# Q-Learning Algorithm
st.header("Roster Results")

with st.spinner("Generating optimal roster..."):
    # Generate Roster and Create DataFrame
    roster = train_roster_agent(shift_config)
    #roster = train_roster_agent(shift_config, availability_lookup)
records = []
for (day, shift), assignments in roster.items():
    for emp_id, svc_id in assignments:
        emp_name = employee_features.loc[employee_features['EmployeeDetailId'] == emp_id, 'FirstName'].values[0]
        svc_name = sample_services.loc[sample_services['ServiceId'] == svc_id, 'ServiceName'].values[0]
        reliability = employee_features.loc[employee_features['EmployeeDetailId'] == emp_id, 'Reliability'].values[0]
        records.append({
            'Day': day + 1,
            'Shift': SHIFT_LABELS.get(shift, f'Shift {shift}'),
            'Employee': emp_name,
            'Reliability': reliability,
            'Service': svc_name
        })

roster_df = pd.DataFrame(records)

# Streamlit UI
st.title("Auto-Rostering with Reinforcement Learning")
st.write("### Generated Roster")
st.dataframe(roster_df)

# Download button
csv = roster_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Roster as CSV",
    data=csv,
    file_name='generated_roster.csv',
    mime='text/csv'
)

# Charts
st.write("### Total Shift Assignments per Employee")
assignment_counts = roster_df['Employee'].value_counts()
st.bar_chart(assignment_counts)

st.write("### Reliability Score Distribution")
fig, ax = plt.subplots()
sns.histplot(roster_df['Reliability'], kde=True, bins=10, ax=ax)
st.pyplot(fig)

st.write("### Service Distribution by Shift")
fig2, ax2 = plt.subplots()
sns.countplot(data=roster_df, x='Shift', hue='Service', ax=ax2)
plt.xticks(rotation=0)
st.pyplot(fig2)
