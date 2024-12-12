from utilis import *
from sklearn.model_selection import train_test_split

path = 'Data'
data = importDataInfo(path)

balanceData(data, display=False)

imgsList, steeringList = loadData(path, data)

imgTrain, imgVal, steerTrain, steerVal = train_test_split(imgsList, steeringList, test_size=0.2, random_state=5)

model = createModel()
model.summary()

history = model.fit(batchGen(imgTrain, steerTrain, 40, 1),
                    steps_per_epoch=2000,
                    epochs=10,
                    validation_data=batchGen(imgVal, steerVal, 40, 0),
                    validation_steps=500)

# history = model.fit(imgTrain, steerTrain,
#                     epochs=15, batch_size=100, shuffle=True,
#                     validation_data=(imgVal, steerVal))


model.save('models/model_moreData.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()