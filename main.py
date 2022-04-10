from knn import KNNClassifier
from svr import SVR
from rand_model import rand_select
from data_process import dataset
from utils import accuracy
from sklearn import ensemble, neural_network, preprocessing, neighbors
from sklearn.metrics import mean_squared_error

model_set = {'myknn':KNNClassifier, 
             'knn':neighbors.KNeighborsRegressor,
            'rfr':ensemble.RandomForestRegressor,
            'mlp':neural_network.MLPRegressor,
            'svr':SVR,
            'rand':rand_select}

if __name__ == '__main__':

    # define
    model_name = 'knn'
    use_normalize = True
    
    # load data
    train_data = dataset(name='Zigbee', flag='train') # ['Zigbee', 'BLE', 'WiFi']
    test_data = dataset(name='Zigbee', flag='test') # ['Zigbee', 'BLE', 'WiFi']
    
    # data scale
    if use_normalize:
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
    
    # create model
    if model_name == 'rfr':
        model = model_set[model_name](n_estimators=300)
    elif model_name == 'myknn':
        model = model_set[model_name](4, 'weight')
    elif model_name == 'knn':
        model = model_set[model_name](
            n_neighbors = 4, weights='uniform', metric = 'euclidean')
    elif model_name == 'mlp':
        # activation:['identity', 'logistic', 'relu', 'softmax', 'tanh']
        model = model_set[model_name](
            hidden_layer_sizes=(3, 3, 3),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
            random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif model_name == 'rand':
        model = model_set[model_name]('gaussian')
    elif model_name == 'svr':
        model = model_set[model_name](kernel='rbf',C=10, gamma = 0.01)
    
    model.fit(train_x, train_y)
    y_predict=model.predict(test_x)
    if use_normalize:
        y_result = scaler_y.inverse_transform(y_predict)
    else:
        y_result = y_predict.copy()
    
    print(test_data['output_data'])
    print(y_result)
    print("平均误差距离:", accuracy(test_data['output_data'], y_result))
    
    mse = mean_squared_error(y_result, test_data['output_data'])
    print("mean_squared_error: ", mse)