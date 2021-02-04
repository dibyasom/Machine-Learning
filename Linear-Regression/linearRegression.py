import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class linear_reg:
    
    def __init__(self, fileName, Xcoloumn, Ycoloumn, title, estReq=False): #Initializes all the necessary variables to keep track of model.
        self.estimationRequired = estReq
        self.title    = title
        self.dataset  = pd.read_csv(fileName)
        self.Xcoloumn = Xcoloumn
        self.Ycoloumn = Ycoloumn
        self.yPred    = []
        self.bestFit  = {}
        self.X        = np.array([])
        self.Y        = np.array([])
        self.X_test   = np.array([])
        self.Y_test   = np.array([])
        self.yPred_all= np.array([])
        self.acc      = 0
        self.rmse     = 0

    def preprocess(self): #Substitutes missing data in Independent column, making it ready to be used by model 
        self.dataset[self.Xcoloumn].fillna(value=self.dataset[self.Xcoloumn].mean(), inplace=True)
    
    def split(self): #Splits the dataset for training and testing purpose
        splitAt = len(self.dataset)*7//10
        self.X = np.array(self.dataset[self.Xcoloumn][:splitAt])
        self.Y = np.array(self.dataset[self.Ycoloumn][:splitAt])
        self.X_test = np.array(self.dataset[self.Xcoloumn][splitAt:])
        self.Y_test = np.array(self.dataset[self.Ycoloumn][splitAt:])

    def apply_linear_regression(self): #Applies linear regression and fetches the best-fit eq parameters in dictionary.
        x_mean, y_mean = np.mean(self.X), np.mean(self.Y)
        x_x, y_y = self.X-x_mean, self.Y-y_mean
        x_ssd = np.array(np.square(x_x))
        yDiff_xDiff = np.array(np.multiply(x_x, y_y))
        coefficient = np.sum(yDiff_xDiff) / np.sum(x_ssd)
        intercept   = y_mean - coefficient*x_mean
        self.bestFit = {'coefficient': coefficient, 'intercept': intercept}

    def generate_yPred(self): #Generates predicted-Y using best-fit eq generated via linear regression.
        self.yPred = [(self.bestFit['coefficient']*x+self.bestFit['intercept']) for x in self.X_test]
        self.yPred_all = [(self.bestFit['coefficient']*x+self.bestFit['intercept']) for x in self.X]

    def calculate_accuracy(self): #Calculates accuracy of prediction of the model.
        self.rmse = math.sqrt(np.square(np.subtract(self.Y_test,self.yPred)).mean())/self.Y_test.mean()*100
        self.acc  = (100-self.rmse)

    def trainAndTest(self):  #Calls all necessary functions in proper heirarchy for smooth control flow and convenience, hehe:)
        self.split()
        self.apply_linear_regression()
        self.generate_yPred()
        self.calculate_accuracy()

def visualize(packet): #Visualizes the linear model, with regression line, predicted data-points, given data-points, and noise cloud.
    packet = [packet[0]]
    fig, ax = plt.subplots()
    for plot_id, self in enumerate(packet):
        keyContent = 'Accuracy: {:.2f}%'.format(self.acc)+self.title
        fig.patch.set_facecolor('xkcd:mint green')
        plt.text(0.05, 0.95, keyContent, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='w', alpha=0.5), verticalalignment='top')
        ax.set_facecolor('k')
        plt.plot(self.X_test, self.yPred, color='red')
        plt.plot(self.X, self.yPred_all, color ='red')
        plt.scatter(self.X, self.Y, c='#32CD32', s=40, marker="*", alpha = 0.6)
        

        for x,y,_y in zip(self.X_test, self.Y_test, self.yPred):
            rgb = (np.random.random(), np.random.random(), np.random.random())
            plt.scatter(x, _y, c='b', s=20, marker="o", alpha=0.5)
            plt.scatter(x,  y, c=rgb, s=75, marker="^", alpha=1)
            plt.plot([x, x], [_y, y], color=rgb)

    plt.suptitle('Linear Regression', fontsize=20, bbox=dict(facecolor='m', alpha=0.6))
    plt.tight_layout()
    plt.show()

def main(): #Main funtion, basically the linear-regression applicable dateset should be called as memeber of class linear-reg for model to initiate.
    expSalaryModel  = linear_reg('Resources/salary_data.csv', 'YearsExperience', 'Salary',  '\n*  -TrainData\n^ -PredictedData\nRed | -Regression Line\nRest colored |s - noise/error')
    expSalaryModel.trainAndTest()
    visualize([expSalaryModel]) #Calls visualizer to plot every insight gained from dataset.

if __name__=="__main__":
    main()
