import pandas as pd
import numpy as np
import random
import joblib
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def df_trim(df):
    out = df.copy()

    # remove rows whose RESULT is in this list
    out = out[~out['RESULT'].isin(['Timeout', 'Penalty', 'Interception'])]

    # drop rows with missing OFF FORM or PERSONNEL
    out = out[out['OFF FORM'].notna() & out['PERSONNEL'].notna()]

    return out

def create_features(df):
    df_out = df.copy()
    df_out['GN/LS LAG 1'] = df_out['GN/LS'].shift(1)
    df_out['GN/LS LAG 2'] = df_out['GN/LS'].shift(2)

    df_out['PLAY # LAG 1'] = df_out['PLAY #'].shift(1)
    df_out['PLAY # LAG 2'] = df_out['PLAY #'].shift(2)

    df_out['PLAY TYPE LAG 1'] = df_out['PLAY TYPE'].shift(1)
    df_out['PLAY TYPE LAG 2'] = df_out['PLAY TYPE'].shift(2)

    df_out.loc[df_out['PLAY #'] - df_out['PLAY # LAG 1'] != 1, 'GN/LS LAG 1'] = 0
    df_out.loc[df_out['PLAY #'] - df_out['PLAY # LAG 2'] != 2, 'GN/LS LAG 2'] = 0

    df_out['OFF FORM LAG 1'] = df_out['OFF FORM'].shift(1)
    df_out['OFF FORM LAG 2'] = df_out['OFF FORM'].shift(2)

    df_out.loc[df_out['PLAY #'] - df_out['PLAY # LAG 1'] != 1, 'OFF FORM LAG 1'] = 'NONE'
    df_out.loc[df_out['PLAY #'] - df_out['PLAY # LAG 2'] != 2, 'OFF FORM LAG 2'] = 'NONE'

    df_out['TIME TO HALF'] = (df_out['TIME TO HALF'].astype(str).str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1])))

    df_out['SCORE DIFF'] = df_out['OPP SCORE'] - df_out['OWN SCORE']

    df_out['HASH'] = df_out['HASH'].astype(str).str.strip()

    df_out['MID OR NOT'] = df_out['HASH'] == 'M'
    df_out['HASH OR NOT'] = df_out['HASH'] != 'M'

    df_out['0-2'] = df_out['DIST'] <= 2
    df_out['2-6'] = (df_out['DIST'] > 2) & (df_out['DIST'] <= 6)
    df_out['6+'] = df_out['DIST'] > 6

    # df['PERSONNEL!'] = df['PERSONNEL'].astype(str).str.strip()
    # df = pd.get_dummies(df, columns=['PERSONNEL'])

    print("Before OHE:", df_out['PERSONNEL'].unique())

    # Reset index before doing anything
    df_out = df_out.reset_index(drop=True)

    # Create dummy variables in a single clean step
    dummies = pd.get_dummies(
        df_out['PERSONNEL'].astype(str).str.strip(),
        prefix='PERSONNEL'
    )

    # Concatenate and then drop the original column
    df_out = pd.concat([df_out, dummies], axis=1).drop(columns=['PERSONNEL'])
    
    print("After OHE columns:", df_out.filter(like='PERSONNEL_').columns.tolist())
    print(df_out.filter(like='PERSONNEL_').tail(3))

    df_out['PLAY OF DRIVE NUM'] = -1
    for i in range(len(df)):
        if (df_out.iloc[i]['DN'] == 0):
            df_out.iloc[i, df_out.columns.get_loc('PLAY OF DRIVE NUM')] = 0
        else:
            df_out.iloc[i, df_out.columns.get_loc('PLAY OF DRIVE NUM')] = df_out.iloc[(i - 1), df_out.columns.get_loc('PLAY OF DRIVE NUM')] + 1

    # Loop through all columns that start with "PERSONNEL_"
    for col in df_out.columns:
        if col.startswith("PERSONNEL_"):
            df_out[f"{col} LAG 1"] = df_out[col].shift(1)
            df_out[f"{col} LAG 2"] = df_out[col].shift(2)

    df_out['2 MIN'] = df_out['2 MIN'].astype(str).str.strip()
    df_out['2 MIN OR NOT'] = -1
    df_out['2 MIN OR NOT'] = df_out['2 MIN'].apply(lambda x: 1 if x == "Y" else 0)

    df_out['OWN END'] = -1
    df_out['OWN END'] = df_out['YARD LN'].apply(lambda x: 1 if x < 0 else 0)

    df_out['OPP END'] = -1
    df_out['OPP END'] = df_out['YARD LN'].apply(lambda x: 1 if x >= 0 else 0)

    df_out['RED ZONE'] = -1
    df_out['RED ZONE'] = df_out['YARD LN'].apply(lambda x: 1 if (x <= 20 and x > 0) else 0)
    df_out['HALF_NUM'] = -1
    df_out['HALF_NUM'] = df_out['QTR'].apply(lambda x: 1 if x <= 2 else 2)

    df_out['TIME LEFT'] = -1
    df_out['TIME LEFT'] = df_out.apply(lambda row: row['TIME TO HALF'] if row['HALF_NUM'] == 2 else (row['TIME TO HALF'] + 900), axis=1)

    df_out['PPS NEEDED'] = -1
    df_out['PPS NEEDED'] = (df_out['SCORE DIFF']) * -1/df_out['TIME LEFT']

    df_out['WINNING'] = -1
    df_out['WINNING'] = df_out['SCORE DIFF'].apply(lambda x: 1 if x < 0 else 0)

    df_out['DN X DIST'] = -1
    df_out['DN X DIST'] = df_out['DN'] * df_out['DIST']

    df_out['PREV PLAY PASS OR NOT'] = -1
    df_out['PREV PLAY PASS OR NOT'] = df_out['PLAY TYPE LAG 1'].apply(lambda x: 1 if x == "Pass" else 0)

    df_out['SCORE DIFF ^2'] = df_out['SCORE DIFF'] * df_out['SCORE DIFF']
    df_out['SCORE DIFF ^2'] = df_out['WINNING'].apply(lambda x: -x if x > 0 else x)

    df_out['SCORE DIFF x TIME LEFT'] = df_out['SCORE DIFF'] * df_out['TIME LEFT']

    df_out['SCORE DIFF x DN'] = df_out['SCORE DIFF'] * df_out['DN']

    df_out['SCORE DIFF / 7'] = df_out['SCORE DIFF'] / 7
    df_out['TIME LEFT * SCORE DIFF / 7'] = df_out['TIME LEFT'] * df_out['SCORE DIFF'] / 7

    df_out['SCORE DIFF x QTR'] = df_out['SCORE DIFF'] * df_out['QTR']

    df_out['YARDS TO TD'] = df_out['YARD LN'].apply(lambda x: x + 100 if x < 0 else x)
    df_out['YARDS TO TD * SCORE DIFF / 7'] = df_out['YARDS TO TD'] * df_out['SCORE DIFF'] / 7

    # drop the first 2 plays (because of lag features)
    df_out = df_out.iloc[2:].reset_index(drop=True)

    return df_out

def split_train(df, location, inputs, target):
    split_index = int(len(df) * location)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train[inputs]
    X_test = test[inputs]
    y_train = train[target]
    y_test = test[target]

    return X_train, X_test, y_train, y_test

# TEMP

# Copy df, remove weird plays, create features
def pre_split_prep(df):
    df = df.copy()
    df = df_trim(df)
    df = create_features(df)
    return df

def edit_previous_play(df, values=None, rewrite='y'):
    df_new = df.copy()
    if (rewrite =='y'):
        if (values is None):
            return "Rewriting requires a list of values"
        df_new.loc[len(df_new) - 1] = values

    elif (rewrite =='n'):
        df_new = df_new.drop(index=len(df_new) - 1)

    else:
        return "Must set rewrite='y' or 'n'"

    return df_new

def create_possible_rows(df_historical, values):
    # possibilities = []

    # Verify 'PERSONNEL' column exists
    if 'PERSONNEL' not in df_historical.columns:
        raise KeyError("The dataframe must contain a 'PERSONNEL' column.")

    # Get all unique personnel options
    options = df_historical['PERSONNEL'].dropna().unique().tolist()
    possible_rows = []

    for option in options:
        new_row = values.copy()
        new_row['PERSONNEL'] = option

        # Append as a clean dict matching df_historical.columns
        possible_rows.append({
            col: new_row.get(col, None) for col in df_historical.columns
        })
        # print(new_row)

    return possible_rows    

def prompt_new_play_inputs(
    df_history,
    play_num,
    qtr,
    time_to_half,
    dn,
    dist,
    yard_ln,
    hash_opt,
    own_score,
    opp_score,
    two_min_flag
):
    """
    Build and return a play_dict using the provided field values.
    No globals are referenced here — all inputs are passed from the frontend.
    """

    play_dict = {
        "PLAY #": play_num,
        "QTR": qtr,
        "TIME TO HALF": time_to_half,
        "DN": dn,
        "DIST": dist,
        "YARD LN": yard_ln,
        "HASH": hash_opt,
        "PLAY TYPE": None,        # set later in Enter Actual Results
        "RESULT": None,           # set later
        "OWN SCORE": own_score,
        "OPP SCORE": opp_score,
        "GN/LS": None,            # set after play
        "2 MIN": two_min_flag,
        "PERSONNEL": None,
        "OFF FORM": None,
    }

    messagebox.showinfo("Inputs Captured", f"Play Info: {play_dict}")
    return play_dict




def predict_top2(model, X):
    y_proba = model.predict_proba(X)
    top2_idx = np.argsort(y_proba, axis=1)[:, -2:]
    classes = model.named_steps['classifier'].classes_
    return np.array([[classes[i] for i in row] for row in top2_idx])

# For top1 prediction simply do the following: y_pred = model_top1.predict(new_X)

# I think there will be extra columns that need to be removed at the X = ... section, must adress later
# I think this is solved


def display_forms(df_hist, current_play, model):
    predictions = {}

    possible_rows = create_possible_rows(df_hist, current_play)

    for row in possible_rows:

        # print("first check: did each possibility get appended as the last row, the following should be the new row last")
        # print(df_hist["PERSONNEL"].tail(5))
        
        # Do preprocessing
        df_temp = df_trim(df_hist)

        # Create temp df: existing + new row
        df_temp = pd.concat([df_temp, pd.DataFrame([row])], ignore_index=True)

        # Do feature engineering
        df_temp = create_features(df_temp)

        # print("check after pre_split_prep: next prints are the last 5 columns of each personnel OHE column in order 10, 11, 12, 11T, 12T")
        # print(df_temp["PERSONNEL_10"].tail(5))
        # print(df_temp["PERSONNEL_11"].tail(5))
        # print(df_temp["PERSONNEL_12"].tail(5))
        # print(df_temp["PERSONNEL_11T"].tail(5))
        # print(df_temp["PERSONNEL_12T"].tail(5))

        # Remove the target variable from the set to select the inputs only
        X = df_temp.iloc[[-1]].drop(columns=['OFF FORM'])

        # Get top 2 most likely formations for this personnel group
        top2_preds = predict_top2(model, X)

        # Store in a dictionary using the personnel group as the key for ease of use with GUI later
        personnel_group = row['PERSONNEL']
        predictions[personnel_group] = top2_preds.flatten().tolist()

    return predictions

def enter_results(df, results_dict):
    """
    Update the last play in df_history with the entered results.
    results_dict should include:
        'PLAY TYPE', 'RESULT', 'GN/LS', 'PERSONNEL', 'OFF FORM'
    """
    df_new = df.copy()
    last_idx = len(df_new) - 1

    # Loop through provided updates and apply them
    for col, val in results_dict.items():
        if col in df_new.columns:
            df_new.at[last_idx, col] = val

    messagebox.showinfo("Results Recorded", f"Updated play #{df_new.at[last_idx, 'PLAY #']} with results.")
    return df_new




