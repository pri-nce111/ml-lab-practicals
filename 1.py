# Student details
NAME = "Prince"
REG_NO = "2502030029"
 
import sys
import numpy as np
import pandas as pd
import matplotlib
import sklearn
 
# Python basics: variables, list, dictionary, function
marks = [78, 85, 92, 67, 88]
student = {"name": NAME, "reg_no": REG_NO, "marks": marks}
 
def average(values):
    """Return the average of a list of numbers."""
    return sum(values) / len(values)
 
print("Python version      :", sys.version.split() [0])
print("NumPy version       :", np.__version__)
print("Pandas version      :", pd.__version__)
print("Matplotlib version  :", matplotlib.__version__)
print("Scikit-learn version:", sklearn.__version__)
print()
print("Student record      :", student)
print("Average marks       :", average(marks))
print("\n" + "-" * 50)
print(f"Name: {NAME} | Register No: {REG_NO}")
