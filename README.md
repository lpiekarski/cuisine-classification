# cuisine classification program
Program predicting the cuisine of the recipe given the ingredients list. <br/>

Run the cuisine_classification.py script with options:

| option | long option | argument | description                                                                                                                                                                                                                                                 |
|--------|-------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -p     | --predict   | [FILE]   | predicts a cuisine given the json file <br/> consisting of list of dictionaries containing key: <br/> "ingredients" - list of ingredients <br/> "id" - recipe id<br/> the result is written into the file specified by <br/> -out-file and is in csv format |
| -h     |  --help     |          |  shows this message                                                                                                                                                                                                                                          |
| -t     |  --train    | [FILE]   |  trains the neural network with the new data <br/> [FILE] format should be json list of dictionaries <br/> containing keys: <br/> "cuisine" - string <br/> "ingredients" - list of ingredients<br/> default file path is "./input/train.json"                |
| -e     |  --epochs   | [NUM]    | changes the number of epochs executed during training <br/> default is 10                                                                                                                                                                                    |
| -c     |  --cuisines |          |  shows list of cuisines that the program is able to identify                                                                                                                                                                                                |
| -o     |  --out-file |  [FILE]  |  specifies the out file name for predictions                                                                                                                                                                                                                |

Sample train file: [input/train.json](https://github.com/lpiekarski/cuisine-classification/blob/master/input/train.json) <br/>
Sample prediction file: [input/test.json](https://github.com/lpiekarski/cuisine-classification/blob/master/input/test.json) <br/>
