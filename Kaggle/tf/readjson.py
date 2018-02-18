import json

lr = [0.5, 0.2, 0.1, 0.05]
batch = [1000, 100, 10]
epochs = [100, 300, 500]

for l in lr:
  for b in batch:
    for e in epochs:
      json_d = open('perceptron4-'+str(l)+'-'+str(b)+'-'+str(e)).read()
      data = json.loads(json_d)

      print('perceptron-'+str(l)+'-'+str(b)+'-'+str(e))
      for i in range(5):
        name = 'perceptron'+str(i)+'-'+str(l)+'-'+str(b)+'-'+str(e)      
        print('recall: {} precision: {} specificity: {} accuracy: {}'.format(data[name][0][1], data[name][0][2], data[name][0][3], data[name][0][4]))
      print()
