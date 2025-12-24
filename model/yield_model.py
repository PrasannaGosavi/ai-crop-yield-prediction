import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.DataFrame({
    "health":[0.4,0.6,0.8,0.9],
    "rainfall":[600,800,1000,1200],
    "temperature":[25,27,29,31],
    "soil":[1,2,3,1],
    "yield":[2200,3000,3800,4200]
})

X = data[["health","rainfall","temperature","soil"]]
y = data["yield"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X,y)

pickle.dump(model, open("yield_model.pkl","wb"))
print("Yield model saved")
