# Cuisine classification program
Program predicting the cuisine of the recipe given the ingredients list.

## Run notes
Run the ```cuisine_classification.py``` script with options:

| option | long option | argument | description                                                                                                                                                                                                                                                 |
|--------|-------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -p     | --predict   | [FILE]   | predicts a cuisine given the json file <br/> consisting of list of dictionaries containing key: <br/> _"ingredients"_ - list of ingredients <br/> _"id"_ - recipe id<br/> the result is written into the file specified by <br/> _-out-file_ and is in _.csv_ format |
| -h     |  --help     |          |  shows this message                                                                                                                                                                                                                                          |
| -t     |  --train    | [FILE]   |  trains the neural network with the new data <br/> _[FILE]_ format should be _.json_ list of dictionaries <br/> containing keys: <br/> _"cuisine"_ - string <br/> _"ingredients"_ - list of ingredients<br/> default file path is ```./input/train.json```                |
| -e     |  --epochs   | [NUM]    | changes the number of epochs executed during training <br/> default is _10_                                                                                                                                                                                    |
| -c     |  --cuisines |          |  shows list of cuisines that the program is able to identify                                                                                                                                                                                                |
| -o     |  --out-file |  [FILE]  |  specifies the out file name for predictions                                                                                                                                                                                                                |

## Training data examples

* Sample _train_ file: [input/train.json](https://github.com/lpiekarski/cuisine-classification/blob/master/input/train.json)
* Sample _prediction_ file: [input/test.json](https://github.com/lpiekarski/cuisine-classification/blob/master/input/test.json)

## Additional informations

Current accuracy is _~77%_ but can go up to _~96%_ with more train time.
