# fMRI Face vs House Decoder
Decodes brain activity from the Nilearn Haxby 2001 dataset - dectecting whether someone is looking at a face or a house based on their fMRI scan.

# What it does
- Downloads the Haxby dataset through Nilearn
- Loads the brain images
- Reads the labels from a CSV file(space seperated, not commas)
- Filters data to only the 'face' and 'house' trials
- Trains a SVC decoder with standardisation and a brain mask
- Prints out the dataset info, label counts, and cross- validation scores

# Requirements
Python 3 and these:
pip install nilearn numpy pandas

# How to run
python load_data.py

# Dataset
Haxby 2001, pulled through Nilearn's built in fetcher

# Method
Model: Support Vector Classifier(SVC)
Task: Face vs House (Bineary Classification)
Preprocessing: Standardisation, uses the dataset's mask
Labels: Read from CSV with sep=" "

# Output
You'll see:
- Dataset and image shape info
- How many samples per label
- Cross validation scores per label
- Mean cross- validation score

# Notes 
- Only two categories: Face and House
-Label file uses spaces, not commas
- No notebooks, just a script that runs

