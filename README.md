# Fruit-Detection
![yolo-predictions](https://user-images.githubusercontent.com/69467096/113019717-92d6d800-918a-11eb-8556-244cf530767f.png)

:star: Real Time Fruit Detection
### real time fruit identifier
1. Tomato
2. Banana
3. Blueberry
4. Strawberry
5. Corn
6. Crimson-Golden Apple
7. Lemon and Lime
8. Avocado
9. Cherry
10. Raspberry
### It is written in the explanations about the codes
### dataset i use; https://www.kaggle.com/moltean/fruits
you can increase the layers to make the model more accurate
```python
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.2))
```
### and epoch size like;
```python
hist = model.fit_generator(dataGen.flow(x_train,y_train,batch_size=batch_size),
                                        validation_data = (x_validation,
                                                           y_validation),
                                        epochs = 60, <------------------
                                        steps_per_epoch = x_train.shape[0]//batch_size,
                                        shuffle = 1)
```
