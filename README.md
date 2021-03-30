# Fruit-Detection
![shangyu-wang-render02](https://user-images.githubusercontent.com/69467096/113021239-0fb68180-918c-11eb-9fc4-8d760e79853f.jpg)

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
### And zoom_range the higher you raise, the more successful you will predict
```python
dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.6, <-------
                             rotation_range = 10)
```
### If you increase the data set to a higher amount I used 600 photos
### When splitting the data set, you can replace the training data with
```python
x_train, x_test, y_train, y_test = train_test_split(images,classNo,test_size = 0.2, <------ %80 percent for training 
                                                    random_state = 42)
```
