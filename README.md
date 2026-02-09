This project includes files with a pre-processing pipeline and a GUI program that predicts the top 2 most likely offensive formations for each personnel group that team has run in the past given the context of a specific play.
The pipeline includes feature extraction, genetic algorithm for feature selection, and training of a Random Forest.

It is important to note that this project is completely done for the format of the CWRU specific data meaning that it likely would not directly work with a different data format plugged in without some processing to match up the formats first.
For privacy reasons the CWRU data has not been uploaded so it would be difficult for anyone to directly test this program out but there is a demonstration video in the repository called Prediction_demo.mp4 to see the program in action. 

As different iterations were tested and the trained models were being stored and improved for specific teams we would be facing in the future, multiple different files were created.
The most relevant files to viewers are the following:
1. formation_prediction_GUI_v2.ipynb
   The jupyter notebook formation_prediction_GUI_v2.ipynb is where the final versions of functions and pipeline are housed for when a trained model needs to be made for a new team in a new year.
   
3. prediction_gui_backend.py
   The python file prediction_gui_backend.py houses many of the functions that are used to make a prediction.
   
5. prediction_gui.py
   The python file prediction_gui.py contains the code for the buttons of the GUI. When running the prediction model this is the file that is run.
   
7. Prediction_Demo.mp4
   The mp4 video file is a video demonstration of how the program works. 

If you are interested to learn more feel free to reach out to my email shane.e.schwartz32@gmail.com
