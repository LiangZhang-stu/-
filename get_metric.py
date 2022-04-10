import os
from main import *

train_data_path = [os.path.join(os.getcwd(), 'data', 'Database_Scenario1.xlsx'), os.path.join(os.getcwd(), 'data', 'Database_Scenario1_d.xlsx')]
test_data_path = [os.path.join(os.getcwd(), 'data', 'Tests_Scenario1.xlsx'), os.path.join(os.getcwd(), 'data', 'Tests_Scenario1_d.xlsx')]
# train_data_path = os.path.join(os.getcwd(), 'data', 'Database_Scenario1_d.xlsx')
# test_data_path = os.path.join(os.getcwd(), 'data', 'Tests_Scenario1_d.xlsx')
model_name = ['knn', 'rfr', 'mlp', 'svr']
use_normalize = [False, True]

data = ['Zigbee', 'BLE', 'WiFi']

for d in data:
    print(d)
    for train_path, test_path in zip(train_data_path, test_data_path):
        print(train_path.split('/')[-1])
        train_data = dataset(name=d, train_data_path=train_path,
                             test_data_path=test_path, flag='train')
        test_data = dataset(name=d, train_data_path=train_path,
                             test_data_path=test_path, flag='test')
        for u in use_normalize:
            print(u)
            if u:
                scaler_x = preprocessing.StandardScaler().fit(train_data['input_data'])
                scaler_y = preprocessing.StandardScaler().fit(train_data['output_data'])
                train_x = scaler_x.transform(train_data['input_data'])
                train_y = scaler_y.transform(train_data['output_data'])
                test_x = scaler_x.transform(test_data['input_data'])
                test_y = scaler_y.transform(test_data['output_data'])
            else:
                train_x = train_data['input_data']
                train_y = train_data['output_data']
                test_x = test_data['input_data']
                test_y = test_data['output_data']
            
            for m in model_name:
                print(m)
                # create model
                if m == 'rfr':
                    model = model_set[m](n_estimators=300)
                elif m == 'myknn':
                    model = model_set[m](4, 'weight')
                elif m == 'knn':
                    model = model_set[m](
                        n_neighbors = 4, weights='uniform', metric = 'euclidean')
                elif m == 'mlp':
                    # activation:['identity', 'logistic', 'relu', 'softmax', 'tanh']
                    model = model_set[m](
                        hidden_layer_sizes=(3, 3, 3),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                        early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                elif m == 'rand':
                    model = model_set[m]('gaussian')
                elif m == 'svr':
                    model = model_set[m](kernel='rbf',C=10, gamma = 0.01)
                
                model.fit(train_x, train_y)
                y_predict=model.predict(test_x)
                
                if u:
                    y_result = scaler_y.inverse_transform(y_predict)
                else:
                    y_result = y_predict.copy()
                
                # print(test_data['output_data'])
                # print(y_result)
                err, thelta = accuracy(test_data['output_data'], y_result)
                print(f"平均误差距离:{err}, thelta^2:{thelta}")
        