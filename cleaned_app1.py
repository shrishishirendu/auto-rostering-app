
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import streamlit as st
from PIL import Image
import os

# Load and display the logo at the top
logo = Image.open("iSoft logo.png")
st.image(logo, width=300)  # You can adjust the width as needed

st.markdown("<h1 style='text-align: right;'></h1>", unsafe_allow_html=True)



# Load and encode the logo image
#logo_path = "iSoft logo.png"
#with open(logo_path, "rb") as img_file:
#    logo_base64 = base64.b64encode(img_file.read()).decode()

# Display the logo in the sidebar
#st.sidebar.markdown(f"""
#    <div style="text-align: center;">
#        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 80%; max-width: 150px; margin-bottom: 10px;">
#        <h4>Auto Rostering</h4>
#    </div>
#""", unsafe_allow_html=True)


#def load_data():
#    excel_data = pd.ExcelFile("C:\All Working Folders\Autoroastering\OrionCare Data.xlsx")
#    employee_df = excel_data.parse('Tb_EmployeeDetails')
#    service_df = excel_data.parse('Tb_ServiceMaster')
#    return employee_df, service_df

# employee_df, service_df = load_data()



# Sidebar Uploads
# st.sidebar.title("Upload Excel File")
#uploaded_excel = st.sidebar.file_uploader("C:\\All Working Folders\\Autoroastering\\auto-rostering-app\\OrionCare Data.xlsx", type=["xlsx"])

st.sidebar.title("Upload Shift Requirements")
#shift_file = st.sidebar.file_uploader("Upload Shift Requirement File (.csv)", type=["csv"])

shift_file = st.sidebar.file_uploader("Upload Shift Requirement File (.csv)", type=["csv"])

#if shift_file is None:
#    st.title("Auto Rostering Solution")
#    st.info("Please upload a shift requirement CSV file to continue.")
#   st.stop()

if shift_file is None:
    st.set_page_config(page_title="Auto Rostering", layout="centered")
    st.title("Auto Rostering Solution")
    st.info("Please upload a shift requirement CSV file using the sidebar.")
    st.markdown("This tool uses reinforcement learning to allocate staff optimally based on shift requirements.")
    st.markdown("---")
    st.markdown("ℹ️ Waiting for your shift file upload...")
    st.stop()


#if shift_file is None:
#    st.title("Auto Rostering Solution")
#    st.info("Please upload a shift requirement CSV file from the sidebar to continue.")
#    st.stop()
#else:
#    shift_config = pd.read_csv(shift_file, header=None)
#    shift_config.columns = ['Morning Shift', 'Evening Shift', 'Night Shift'][:shift_config.shape[1]]
#    shift_config.index = [f'Day {i+1}' for i in range(shift_config.shape[0])]
#
#    with st.sidebar:
#        st.write("### Uploaded Shift Matrix")
#        st.dataframe(shift_config)
#        st.markdown("**Note:** Numbers indicate how many staff are required for that shift (e.g., 2 = 2 staff)")


# Load and validate input files
#if uploaded_excel is None or shift_file is None:
#    st.warning("Please upload both the Excel and Shift Requirement CSV files to continue.")
#    st.stop()

# Parse files
#excel_data = pd.ExcelFile("C:\All Working Folders\Autoroastering\OrionCare Data.xlsx")
excel_data = pd.ExcelFile("OrionCare Data.xlsx")

employee_df = excel_data.parse('Tb_EmployeeDetails')
service_df = excel_data.parse('Tb_ServiceMaster')
shift_config = pd.read_csv(shift_file, header=None)
shift_config.columns = ['Morning Shift', 'Evening Shift', 'Night Shift'][:shift_config.shape[1]]
shift_config.index = [f'Day {i+1}' for i in range(shift_config.shape[0])]
NUM_DAYS, NUM_SHIFTS = shift_config.shape

# Preview shift matrix
with st.sidebar:
    st.write("### Uploaded Shift Matrix")
    st.dataframe(shift_config)
    st.markdown("**Note:** Numbers indicate how many staff are required for that shift (e.g., 2 = 2 staff)")

# Prepare employee data
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
    def __init__(self, shift_config, enforce_no_consecutive=True):
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

# Q-learning Agent
def train_roster_agent(shift_config, episodes=1000):
    Q = defaultdict(float)
    enforce_no_consecutive = st.sidebar.checkbox("Disallow Consecutive Shifts", value=True)
    env = RosterEnv(shift_config, enforce_no_consecutive=enforce_no_consecutive)
    alpha, gamma, epsilon = 0.1, 0.95, 0.2

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.get_available_actions()
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_vals = [Q[(state, a)] for a in actions]
                max_q = max(q_vals)
                best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]
                action = random.choice(best_actions)
            next_state, reward, done = env.step(action)
            max_future_q = max([Q[(next_state, a)] for a in actions], default=0)
            Q[(state, action)] += alpha * (reward + gamma * max_future_q - Q[(state, action)])
            state = next_state
    return env.roster

# Run Agent
st.header("Roster Results")
with st.spinner("Generating optimal roster..."):
    roster = train_roster_agent(shift_config)

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

# Display Results
st.title("Auto-Roster Loading...")
st.dataframe(roster_df)

csv = roster_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Roster as CSV", data=csv, file_name="generated_roster.csv", mime="text/csv")

st.write("### Total Shift Assignments per Employee")
st.bar_chart(roster_df['Employee'].value_counts())

st.write("### Reliability Score Distribution")
fig, ax = plt.subplots()
sns.histplot(roster_df['Reliability'], kde=True, bins=10, ax=ax)
st.pyplot(fig)

st.write("### Service Distribution by Shift")
fig2, ax2 = plt.subplots()
sns.countplot(data=roster_df, x='Shift', hue='Service', ax=ax2)
st.pyplot(fig2)
