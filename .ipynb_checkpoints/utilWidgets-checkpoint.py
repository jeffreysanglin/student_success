import matplotlib.pyplot as plt
import seaborn as sns
from numpy import abs
import pandas as pd
import warnings
import ipywidgets as widgets
from IPython.display import display
from scipy.stats import skew, kurtosis


def classifyDistribution(data, column):
    """Return a label for the distrubtion of a data set (expects a DataFrame and column).
    
    Returns one of: Normal, Right Skewed, Left Skewed, leptokurtic, platykurtic
    
    """
    sk = skew(data[column])
    kurt = kurtosis(data[column])
    
    #Check for Skew (the direction of the tail: left, right, or normal)
    if abs(sk) < 0.5:
        distribution_label = 'Normal'
    elif sk > 0:
        distribution_label = 'Right Skewed'
    else:
        distribution_label = 'Left Skewed'
        
    #Check for kurtosis (fatness of the tails)
    if distribution_label == 'Normal':
        if kurt > 3.5:
            distribution_label = 'leptokurtic'
        elif kurt < 2.5:
            distribution_label = 'platykurtic'
        else:
            distribution_label = "Normal"
    return(distribution_label)       


class utilWidgets:
    def __init__(self):
        self.distribution_choices = {}
        
    # Function to display widgets for marking distribution type
    def mark_distribution(self, column, data, infer_labels=True):
        if infer_labels:
            inferred_label = classifyDistribution(data, column)
            self.distribution_choices[column] = inferred_label
        
        # Creating the widget buttons for marking distribution type
        button_normal = widgets.Button(description="Normal")
        button_right = widgets.Button(description="Right Skewed")
        button_left = widgets.Button(description="Left Skewed")
        button_other = widgets.Button(description="Other")

        # Define the callback functions for each button
        def on_button_normal_clicked(b):
            self.distribution_choices[column] = 'Normal'
            print(f"Marked {column} as Normal Distribution")

        def on_button_right_clicked(b):
            self.distribution_choices[column] = 'Right Skewed'
            print(f"Marked {column} as Right Skewed Distribution")
            
        def on_button_left_clicked(b):
            self.distribution_choices[column] = 'Left Skewed'
            print(f"Marked {column} as Left Skewed Distribution")

        def on_button_other_clicked(b):
            self.distribution_choices[column] = 'Other'
            print(f"Marked {column} as Other Distribution")

        # Assign the button functions to be triggered on click
        button_normal.on_click(on_button_normal_clicked)
        button_right.on_click(on_button_right_clicked)
        button_left.on_click(on_button_left_clicked)
        button_other.on_click(on_button_other_clicked)

        buttons = widgets.HBox([button_normal, button_right, button_left, button_other])

        # Display the buttons
        display(buttons)
        return(self)