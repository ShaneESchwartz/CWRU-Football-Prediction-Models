import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import joblib
from prediction_gui_backend import (
    prompt_new_play_inputs,
    display_forms,
    enter_results,
    edit_previous_play,
)

df_history = None
current_play = None
model = None
current_quarter = 1
own_score = 0
opp_score = 0
yard_ln = 0
current_down = 0
current_distance = 10
hash_opt = "M"
two_min_flag = 0
current_play_num = 1
time_to_half = "30:00"

def load_data():
    global df_history
    path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if path:
        df_history = pd.read_excel(path)
        messagebox.showinfo("Loaded", f"Loaded {len(df_history)} plays.")

def load_model():
    global model
    path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pkl")])
    if path:
        model = joblib.load(path)
        messagebox.showinfo("Loaded", "Model loaded successfully!")

def convert_clock_to_half(clock_str, qtr):
    """
    Convert scoreboard time (MM:SS) into TIME TO HALF format.
    """
    mm, ss = map(int, clock_str.split(":"))
    total_seconds = mm * 60 + ss

    if qtr in [1, 3]:
        # 15 minutes left in quarter -> add full quarter length
        time_to_half_seconds = total_seconds + 900
    else:
        # Q2 or Q4 → already counting directly to half/end
        time_to_half_seconds = total_seconds

    # Convert seconds back to MM:SS
    mm2 = time_to_half_seconds // 60
    ss2 = time_to_half_seconds % 60
    return f"{mm2:02d}:{ss2:02d}"


def check_play():
    global df_history, current_play, current_play_num, current_quarter
    global time_to_half, current_down, current_distance, yard_ln
    global hash_opt, own_score, opp_score, two_min_flag

    if df_history is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    # # Increment play number for each new play
    # if current_play_num != 0:
    #     current_play_num = current_play_num + 1

    # Auto-update 2-minute warning flag
    mm, ss = map(int, time_to_half.split(":"))
    total_seconds = mm * 60 + ss
    two_min_flag = "Y" if current_quarter in [2, 4] and total_seconds <= 120 else 0

    # Build play dict using the backend function
    current_play = prompt_new_play_inputs(
        df_history=df_history,
        play_num=current_play_num,
        qtr=current_quarter,
        time_to_half=time_to_half,
        dn=current_down,
        dist=current_distance,
        yard_ln=yard_ln,
        hash_opt=hash_opt,
        own_score=own_score,
        opp_score=opp_score,
        two_min_flag=two_min_flag,
    )

def predict_play():
    global df_history, current_play, model
    global current_play_num, current_quarter
    global time_to_half, current_down, current_distance, yard_ln
    global hash_opt, own_score, opp_score, two_min_flag



    # Auto-update 2-minute warning flag
    mm, ss = map(int, time_to_half.split(":"))
    total_seconds = mm * 60 + ss
    two_min_flag = "Y" if current_quarter in [2, 4] and total_seconds <= 120 else 0

    # Build play dict using the backend function
    current_play = prompt_new_play_inputs(
        df_history=df_history,
        play_num=current_play_num,
        qtr=current_quarter,
        time_to_half=time_to_half,
        dn=current_down,
        dist=current_distance,
        yard_ln=yard_ln,
        hash_opt=hash_opt,
        own_score=own_score,
        opp_score=opp_score,
        two_min_flag=two_min_flag,
    )

    df_history = pd.concat([df_history, pd.DataFrame([current_play])], ignore_index=True)

    # Increment play number for each new play
    current_play_num = current_play_num + 1

    if any(v is None for v in [df_history, current_play, model]):
        messagebox.showerror("Error", "Dataset, model, or inputs missing.")
        return

    predictions = display_forms(df_history, current_play, model)
    show_predictions(predictions)

def show_predictions(preds):
    popup = tk.Toplevel()
    popup.title("Top-2 Predictions")
    tk.Label(popup, text="Predicted Formations", font=("Arial", 14, "bold")).pack(pady=5)
    for p, forms in preds.items():
        tk.Label(popup, text=f"{p}: {forms[0]}, {forms[1]}", font=("Arial", 12)).pack(anchor="w")
    tk.Button(popup, text="OK", command=popup.destroy).pack(pady=5)

# def enter_results_button():
#     global df_history
#     df_history = enter_results(df_history)

def enter_results_button():
    global df_history, time_to_half, hash_opt
    global current_down, current_distance, yard_ln
    global current_quarter

    if df_history is None or len(df_history) == 0:
        messagebox.showerror("Error", "No plays found to update.")
        return

    popup = tk.Toplevel()
    popup.title("Enter Play Results")

    # --- Input fields for each value ---
    # tk.Label(popup, text="Play Type (Run / Pass):").pack(pady=3)
    # play_type_var = tk.StringVar()
    # tk.Entry(popup, textvariable=play_type_var).pack()

    tk.Label(popup, text="Result (e.g., Rush, Complete, Incomplete, Sack, Interception):").pack(pady=3)
    result_var = tk.StringVar()
    tk.Entry(popup, textvariable=result_var).pack()

    tk.Label(popup, text="Yard line the play ended on(-1 to -49 for off team's half, 50 to 1 for def team's half):").pack(pady=3)
    yard_ln_var = tk.StringVar()
    tk.Entry(popup, textvariable=yard_ln_var).pack()

    tk.Label(popup, text="True Personnel (ex: 10, 12, 12T):").pack(pady=3)
    personnel_var = tk.StringVar()
    tk.Entry(popup, textvariable=personnel_var).pack()

    tk.Label(popup, text="Offensive Formation (ex: TRIO, DUCH Y OFF):").pack(pady=3)
    form_var = tk.StringVar()
    tk.Entry(popup, textvariable=form_var).pack()

    tk.Label(popup, text="Hash for next play (ex: L, M, R):").pack(pady=3)
    hash_var = tk.StringVar()
    tk.Entry(popup, textvariable=hash_var).pack()

    tk.Label(popup, text="Time on clock (ex: 12:30):").pack(pady=3)
    time_on_clock_var = tk.StringVar()
    tk.Entry(popup, textvariable=time_on_clock_var).pack()

    def submit():
        global time_to_half, hash_opt, yard_ln, current_distance, current_down, df_history

    # ---- Convert ending yard line ----
        try:
            ending_yard_ln = int(yard_ln_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Ending yard line must be an integer.")
            return

        if (yard_ln > 0 and ending_yard_ln > 0) or (yard_ln < 0 and ending_yard_ln < 0):
            gain_var = abs(ending_yard_ln - yard_ln)

        elif (yard_ln < 0 and ending_yard_ln > 0):
            own_half = 50 - abs(yard_ln)
            opp_half = 50 - ending_yard_ln
            gain_var = own_half + opp_half
        
        elif (yard_ln > 0 and ending_yard_ln < 0):
            own_half = 50 - yard_ln
            opp_half = 50 - abs(ending_yard_ln)
            gain_var = (own_half + opp_half) * -1

        if gain_var >= current_distance:
            current_down = 1
            current_distance = 10

        yard_ln = ending_yard_ln
        user_clock = time_on_clock_var.get().strip()
        time_to_half = convert_clock_to_half(user_clock, current_quarter)

        pass_pos = ['Complete', 'Incomplete', 'Scramble', 'Complete, TD', 'Interception', 'Sack', 'Complete, Fumble']
        run_pos = ['Rush', 'Rush, TD', 'Fumble',]

        result_str = result_var.get().title().strip()
        if result_str in pass_pos:
            play_type = "Pass"
        elif result_str in run_pos:
            play_type = "Run"


        results_dict = {
            "PLAY TYPE": play_type,
            "RESULT": result_var.get().title().strip(),
            "GN/LS": gain_var,
            "PERSONNEL": personnel_var.get().upper().strip(),
            "OFF FORM": form_var.get().upper().strip(),
            "HASH": hash_var.get().upper().strip(),
            "TIME TO HALF": time_to_half,
        }

        # Send to backend function
        from prediction_gui_backend import enter_results
        df_history = enter_results(df_history, results_dict)

        popup.destroy()

    tk.Button(popup, text="Submit", command=submit).pack(pady=8)

def next_quarter():
    global current_quarter
    if current_quarter < 4:
        current_quarter += 1

def new_drive():
    """Start a new possession and set starting yard line."""
    global current_down, current_distance, yard_ln, time_to_half, hash_opt
    current_down = 0           # mark start of possession (first play)
    current_distance = 10      # always start a drive needing 10 yards

    # Create a popup window to get starting yard line
    popup = tk.Toplevel()
    popup.title("New Drive")

    # Text prompt explaining the expected format
    tk.Label(
        popup,
        text="Enter starting yard line:\n(-1 to -49 for off team's half, 50 to 1 for def team's half)",
        wraplength=250
    ).pack(pady=5)

    # Variable to store the user's entry dynamically
    yard_var = tk.DoubleVar()
    tk.Entry(popup, textvariable=yard_var).pack()

    tk.Label(popup, text="Time on clock (ex: 12:30):").pack(pady=3)
    time_on_clock_var = tk.StringVar()
    tk.Entry(popup, textvariable=time_on_clock_var).pack()

    tk.Label(popup, text="Hash for next play (ex: L, M, R):").pack(pady=3)
    hash_var = tk.StringVar()
    tk.Entry(popup, textvariable=hash_var).pack()

    # Function that runs when the user clicks "Start Drive"
    def submit():
        global yard_ln, time_to_half, hash_opt
        val = yard_var.get()
        user_clock = time_on_clock_var.get().strip()
        time_to_half = convert_clock_to_half(user_clock, current_quarter)
        hash_opt = hash_var.get().upper().strip()

        # check that value is valid for either half
        if (-49 <= val <= -1) or (1 <= val <= 50):
            yard_ln = val
            messagebox.showinfo(
                "New Drive",
                f"Drive started at yard line {yard_ln} (1st & 10)."
            )
            popup.destroy()
        else:
            messagebox.showerror(
                "Error",
                "Yard line must be between -49 and -1 or between 1 and 50."
            )

    # Button to confirm and run the submit function
    tk.Button(popup, text="Start Drive", command=submit).pack(pady=5)

def update_score():
    global own_score, opp_score
    # Create a popup window to get scores
    popup = tk.Toplevel()
    popup.title("Update Score")

    tk.Label(popup, text="Your Team's Score:").pack(pady=3)
    own_score_var = tk.IntVar()
    tk.Entry(popup, textvariable=own_score_var).pack()

    tk.Label(popup, text="Opponent's Score:").pack(pady=3)
    opp_score_var = tk.IntVar()
    tk.Entry(popup, textvariable=opp_score_var).pack()

    def submit():
        global own_score, opp_score
        own_score = own_score_var.get()
        opp_score = opp_score_var.get()
        popup.destroy()

    tk.Button(popup, text="Update", command=submit).pack(pady=5)

def edit_last_play():
    global df_history
    df_history = edit_previous_play(df_history)

def save_data():
    global df_history
    if df_history is None:
        messagebox.showerror("Error", "No dataset to save.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv")
    if path:
        df_history.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Saved to {path}")

root = tk.Tk()
root.title("Football Play Predictor")
root.geometry("350x380")

tk.Button(root, text="Load Dataset", width=25, command=load_data).pack(pady=3)
tk.Button(root, text="Load Model", width=25, command=load_model).pack(pady=3)
tk.Button(root, text="Check Current Play", width=25, command=check_play).pack(pady=3)
tk.Button(root, text="Predict Formation", width=25, command=predict_play).pack(pady=3)
tk.Button(root, text="Enter Actual Results", width=25, command=enter_results_button).pack(pady=3)
tk.Button(root, text="Edit Previous Play", width=25, command=edit_last_play).pack(pady=3)
tk.Button(root, text="Next Quarter", width=25, command=next_quarter).pack(pady=3)
tk.Button(root, text="Start New Drive", width=25, command=new_drive).pack(pady=3)
tk.Button(root, text="Update Score", width=25, command=update_score).pack(pady=3)
tk.Button(root, text="Save Dataset", width=25, command=save_data).pack(pady=3)
tk.Button(root, text="Quit", width=25, command=root.quit).pack(pady=10)

root.mainloop()
