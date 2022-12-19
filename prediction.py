import pickle
filename = "loan_match"
loaded_model = pickle.load(open(filename,'rb'))
pred_result = loaded_model.predict([[1,2,75,60,10000000,250000000,60,180,750,8.5,5,780,12500000,180,
213197092,154306057,15784,490202,0,0,688141,490202,0,277156219,1194127,5764941,-420544.59,534227.23,1006311.31,150000000]])
print(pred_result)