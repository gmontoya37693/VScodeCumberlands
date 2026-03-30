
# =============================================================
# header
# Resident Liability Waiver Program Rollover Simulation
#
# Objective:
#   Simulate the rollover of a resident liability waiver program
#   across 26 properties from April 2026 (included) for 15 months.
#   Outputs:
#     - Gantt schedule by month for properties
#     - Accumulated progression of opt-in waiver residents
#     - Unit properties addressed
#     - Remaining renters insurance residents
#     - Drag (delay/lag in progression)
# Consultant: German Montoya
# Company: Acento Real Estate Partners
# Inputs:
#   - Unit count per property
#   - Initial assumptions chart (to be defined after header)
# =============================================================

import pandas as pd
all_properties = [
    ("Ashland Towne Square", 218),
    ("Burnam Woods", 168),
    ("Camden Hills", 152),
    ("Carlyle Landing Apartments", 172),
    ("Centre Pointe", 151),
    ("Centre at Silver Spring", 241),
    ("Chatham Gardens", 417),
    ("Chelsea Park", 71),
    ("Crystal Woods", 343),
    ("Eaton Square", 416),
    ("Governor Square", 232),
    ("Hamilton Gardens", 75),
    ("Indigo", 346),
    ("Jefferson Hall", 115),
    ("Lakeshore at Hampton Center", 385),
    ("Lakeview", 563),
    ("Linden Park Apartments", 199),
    ("Lockwood", 107),
    ("Northampton Reserve", 593),
    ("Old Orchard", 182),
    ("Park Gardens", 51),
    ("Terrace Green", 97),
    ("The Commons at Cowan Boulevard", 238),
    ("The Grove At Alban", 280),
    ("The Landing at Oyster Point", 508),
    ("The Westwinds", 146),
    ("Towne Crest Apartments", 103),
    ("Townsend Square", 214),
    ("Wellington Woods", 109),
    ("Woodgate Apartments", 258),
]



# Mark legacy, rp_income (excluded), and non_legacy
legacy = {"Carlyle Landing Apartments", "Hamilton Gardens", "Jefferson Hall", "Linden Park Apartments", "Park Gardens", "Terrace Green"}
rp_income = {"Camden Hills", "Crystal Woods", "The Commons at Cowan Boulevard", "Wellington Woods"}
df = pd.DataFrame(all_properties, columns=["Property Name", "Unit Count"])
df.insert(0, "No.", range(1, len(df) + 1))
df["Type"] = df["Property Name"].apply(lambda x: "legacy" if x in legacy else ("rp_income" if x in rp_income else "non_legacy"))



# Print all properties with numbering and sum, formatted for readability
print("\nAll Properties and Unit Counts:")
print("----------------------------------------------")
print(f"{'No.':>3}  {'Property Name':35}  {'Unit Count':>9}")
print("----------------------------------------------")
for _, row in df.iterrows():
    print(f"{row['No.']:>3}  {row['Property Name']:<35}  {row['Unit Count']:>9}")
print("----------------------------------------------")
print(f"{'Total':>39}  {df['Unit Count'].sum():>9}")



# Print summary table with formatting (split of 7150)
summary = df.groupby("Type")["Unit Count"].agg(['count', 'sum']).rename(columns={'count': 'Property Count', 'sum': 'Unit Count'})
print("Summary by Type (Split of 7,150):")
print("---------------------------")
print(f"{'Type':<10}  {'Property Count':>14}  {'Unit Count':>10}")
print("---------------------------")
for idx, row in summary.iterrows():
    print(f"{idx:<10}  {row['Property Count']:>14}  {row['Unit Count']:>10}")
print("---------------------------")
print(f"{'Total':<10}  {summary['Property Count'].sum():>14}  {summary['Unit Count'].sum():>10}")



# Rollout dataset: all properties except rp_income (should sum to 6,308)
df_rollout = df[df["Type"] != "rp_income"].reset_index(drop=True)
print("\nRollout dataset (Approachable properties, 6,308 units):")
print("--------------------------------------------------------------")
print(f"{'No.':>3}  {'Property Name':35}  {'Unit Count':>9}  {'Type':>10}")
print("--------------------------------------------------------------")
for _, row in df_rollout.iterrows():
    print(f"{row['No.']:>3}  {row['Property Name']:<35}  {row['Unit Count']:>9}  {row['Type']:>10}")
print("--------------------------------------------------------------")
print(f"{'Total':>39}  {df_rollout['Unit Count'].sum():>9}")

