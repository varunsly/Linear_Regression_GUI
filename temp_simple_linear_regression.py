import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

from PIL import ImageTk, Image

# data reading using pandas
dataset=pd.read_csv("/home/varun/Desktop/test.csv")

x_axis=dataset['x']
y_axis=dataset['y']

#removing the index as regression model doesn't take index
x= np.array(x_axis).reshape(-1,1)
y= np.array(y_axis).reshape(-1,1)

#Using tkinter
root=tk.Tk()
root.title("Regression_GUI")
#root.configure(bg='beige')


#Setting Windows
canvas=tk.Canvas(root, width= 300, height=300)
#background_image=tk.PhotoImage('/home/varun/Desktop/Stock_image.jpg')
#background_label_1=tk.Label(root,image=background_image)
#background_label_1.place(x=0, y=0, relwidth=1, relheight=1)
canvas.pack()
label_1=tk.Label(root, text="Garphical_Representation",borderwidth=.01, font=("Arial Bold",20), justify='center')
#label_1.configure(bg='beige')
canvas.create_window(200,40,window=label_1)

#using regression algorithm impoeted from sklearn
model=LinearRegression()
model.fit(x,y)

regression_model_mse=mean_squared_error(x,y)
print("MSE:", math.sqrt(regression_model_mse))
print("RSE: ", model.score(x,y))
#Setting labels for intercept and coefficient

coefficient=('Value of Co-efficient for the given sample is: ', model.coef_[0])
intercept=('Value of intercept for the given sample is :', model.intercept_[0])

label_intercept=tk.Label(root,text=intercept, justify='center')
label_intercept.configure(bg='beige')
canvas.create_window(260,220, window=label_intercept)
label_coefficient=tk.Label(root, text=coefficient, justify='center')
#label_coefficient.configure(bg='beige')
canvas.create_window(280,240,window=label_coefficient)

#label for input

label_2=tk.Label(root,text="Value for x", justify='center')
#label_2.configure(bg='beige')
canvas.create_window(100,100,window=label_2)

entry_1=tk.Entry(root)
canvas.create_window(270,100,window=entry_1)

#Invoke below comment if delaing with multivariable linear regression
#label_3=tk.Label(root,text="Value of y", justify='center')
#label_3.configure(bg='beige')
#canvas.create_window(120,120,window=label_3)

#entry_2=tk.Entry(root)
#canvas.create_window(270,120,window=entry_2)

#function for accepting value from user and making predictionby the given data
def value():
    global New_X_Value
    New_X_Value=float(entry_1.get())
    
    #global New_Y_Value
    #New_Y_Value=float(entry_2.get())
    
    global res
    res=np.array(New_X_Value).reshape(-1,1)
    
    predicition_result= ('Prediction_Of_Value_of_Y:', model.predict(res))
    label_Prediction=tk.Label(root, text=predicition_result,bg='orange')
    canvas.create_window(260,280, window=label_Prediction)
    
button_prediction=tk.Button(root,text='Predicted Value',command=value,borderwidth=.01,bg='orange')
button_prediction.configure(bg='beige')

canvas.create_window(270,150,window=button_prediction)
    
#Plotting Graph on GUI

figure3 = plt.Figure(figsize=(20,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(dataset['x'].astype(float),dataset['y'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend() 
ax3.set_xlabel('Value of X')
ax3.set_title('Axis-Measurment')





plt.scatter(x,y,color="green")
plt.plot(x,model.predict(x),color="black")
plt.title("Simple Linear Regression")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


root.mainloop()