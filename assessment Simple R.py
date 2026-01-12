# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 19:03:14 2026

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

a=pd.read_csv(r"C:/Users/LENOVO/Downloads/Data Set (1)/delivery_time.csv")
a.columns = a.columns.str.strip().str.replace(' ', '_')
a.sort_values("Sorting_Time",inplace=True)
a.reset_index(inplace=True,drop=True)

X=pd.DataFrame(a["Sorting_Time"])
Y=pd.DataFrame(a["Delivery_Time"])


a.plot(kind = 'box', subplots = True, sharey = False, figsize = (8, 6))
plt.subplots_adjust(wspace = 0.75)  
plt.show()

a.corr()
a.columns = a.columns.str.strip().str.replace(' ', '_')

model=smf.ols('Delivery_Time ~ Sorting_Time',data=a).fit()
model.summary()
cr=np.corrcoef(a.Delivery_Time,(a.Sorting_Time))[0, 1]
predict1=model.predict(a["Sorting_Time"])

residual1=a["Delivery_Time"]-predict1
mean1=np.mean(residual1)
print(mean1)


res_squ1=residual1*residual1
mres1=np.mean(res_squ1)
mres1
rmse1 = np.sqrt(mres1)
rmse1

plt.scatter(a.Sorting_Time,a.Delivery_Time) 
plt.plot(a.Sorting_Time, predict1, "r") 
plt.xlabel('Sorting Time')  
plt.ylabel('Delivery Time')  
plt.title('Linear Regression Line (Delivery_Time ~ Sorting_Time)')  
plt.legend(['Observed data', 'fitted line'])  
plt.show()




model1=smf.ols("Delivery_Time~np.log(Sorting_Time)",a).fit()
model1.summary()
predict2=model1.predict(a["Sorting_Time"])

residual2=a["Delivery_Time"]-predict2
mean2=np.mean(residual2)
print(mean2)

c=np.corrcoef(a.Delivery_Time,np.log(a.Sorting_Time))[0, 1]

res_squ2=residual2*residual2
mres2=np.mean(res_squ2)
mres2
rmse2 = np.sqrt(mres2)
rmse2
plt.scatter(a.Sorting_Time,a.Delivery_Time) 
plt.plot(a.Sorting_Time, predict2, "r") 
plt.xlabel('Sorting Time')  
plt.ylabel('Delivery Time')  
plt.title('Linear Regression Line (Delivery_Time ~ np.log(Sorting_Time))')  
plt.legend(['Observed data', 'fitted line'])  
plt.show()



#exp


model3=smf.ols("np.log(Delivery_Time)~Sorting_Time",a).fit()
model3.summary()

ec=np.corrcoef(np.log(a.Delivery_Time),(a.Sorting_Time))[0, 1]
predict3=model3.predict(a.Sorting_Time)

predict4=np.exp(predict3)
res3=a.Delivery_Time-predict4
res4=res3*res3
res5=np.mean(res4)
print(res5)
Rmse3=np.sqrt(res5)
Rmse3

plt.scatter(a.Sorting_Time,np.log(a.Delivery_Time)) 
plt.plot(a.Sorting_Time, predict3, "r") 
plt.xlabel('Sorting Time')  
plt.ylabel('Delivery Time')  
plt.title('Linear Regression Line (np.log(Delivery_Time) ~ (Sorting_Time))')  
plt.legend(['Observed data', 'fitted line'])  
plt.show()






#ploynomial
model4=smf.ols("np.log(Delivery_Time)~Sorting_Time+I(Sorting_Time*Sorting_Time)",a).fit()
model4.summary()
ec = np.corrcoef(
    np.log(a.Delivery_Time),
    a.Sorting_Time**2
)[0, 1]
values=model4.predict(a.Sorting_Time)
pred4_at = np.exp(values)  
res9 = a.Delivery_Time - pred4_at 
res_sqr4 = res9 * res9  
mse4 = np.mean(res_sqr4)  
rmse4 = np.sqrt(mse4)  
rmse4 

plt.scatter(a.Sorting_Time,np.log(a.Delivery_Time)) 
plt.plot(a.Sorting_Time, predict4, "r") 
plt.xlabel('Sorting Time')  
plt.ylabel('Delivery Time')  
plt.title('Linear Regression Line (np.log(Delivery_Time) ~ (Sorting_Time))')  
plt.legend(['Observed data', 'fitted line'])  
plt.show()

print("The base model has a RMSE of:", rmse1) 
print("The log-transformed model has a RMSE of:", rmse2)  
print("The exponential-transformed model has a RMSE of:", Rmse3)
print("The ploynomail-transformed model has a RMSE of:", rmse4)


data = {"MODEL": pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE": pd.Series([rmse1, rmse2, Rmse3, rmse4])} 
table_rmse = pd.DataFrame(data)  
table_rmse


#Final model
X = a[['Sorting_Time']]     
Y = a[['Delivery_Time']]     


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=30)
plt.scatter(
    np.log(X_train['Sorting_Time']),
    Y_train['Delivery_Time'],
    label='Train'
)

plt.scatter(
    np.log(X_test['Sorting_Time']),
    Y_test['Delivery_Time'],
    label='Test'
)

plt.xlabel("log(Sorting Time)")
plt.ylabel("Delivery Time")
plt.legend()
plt.show()

train_df = pd.concat([X_train, Y_train], axis=1)
test_df  = pd.concat([X_test, Y_test], axis=1)

Final_model=smf.ols("Delivery_Time~np.log(Sorting_Time)",train_df).fit()
Final_model.summary()

Final_predict=Final_model.predict(test_df)

Final_residual = train_df['Delivery_Time'] - Final_predict

Final_mean=np.mean(Final_residual)
print(Final_mean)


Final_res_squ=Final_residual*Final_residual
Final_mres=np.mean(Final_res_squ)
Final_mres
Final_rmse = np.sqrt(Final_mres)
Final_rmse





x_line = np.linspace(
    a['Sorting_Time'].min(),
    a['Sorting_Time'].max(),
    100
)


x_line_df = pd.DataFrame({'Sorting_Time': x_line})


y_line = Final_model.predict(x_line_df)


plt.scatter(
    test_df['Sorting_Time'],
    test_df['Delivery_Time'],
    color='blue',
    label='Test Data'
)


plt.plot(
    x_line,
    y_line,
    color='red',
    linewidth=3,
    label='Log(Input) Regression Line'
)

plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Log(Input) Regression Model")
plt.legend()
plt.show()









