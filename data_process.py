import numpy as np  # numpy库
import pandas as pd  # 导入pandas
import os


input_key = ['RSSI A', 'RSSI B', 'RSSI C']
output_key = ['x', 'y']

def translate(rssi, name):
    if name == 'Zigbee':
        n, C = 2.935, -50.33
    elif name == 'BLE':
        n, C = 2.271, -75.48
    else:
        n, C = 2.162, -45.73
    d = pow(10, (C + rssi) / (10 * n)) # rssi数据默认取反
    return d


def dataset(name, flag, train_data_path, test_data_path, use_tran=False):
    assert name in ['Zigbee', 'BLE', 'WiFi'], 'unknown data'
    assert flag in  ['train', 'test'], 'error select'
    
    if flag == 'train':data_set = pd.read_excel(train_data_path, sheet_name = name)
    else:data_set = pd.read_excel(test_data_path, sheet_name = name)
    input_data = np.column_stack([data_set[key].copy() for key in input_key])
    output_data = np.column_stack([data_set[key].copy() for key in output_key])
    
    if use_tran:
        return {'input_data':translate(input_data, name), 'output_data':output_data}
    else:
        return {'input_data':input_data, 'output_data':output_data}

if __name__ == '__main__':
    excel_name = {'train':'data/Database_Scenario1_d.xlsx',
                  'test':'data/Tests_Scenario1_d.xlsx'}
    for type in ['train', 'test']:
        writer = pd.ExcelWriter(excel_name[type])
        for name in ['Zigbee', 'BLE', 'WiFi']:
            tmp = dataset(name, type, True)
            excel_dict = {input_key[i]:tmp['input_data'][:, i] for i in range(len(input_key))}
            excel_dict.update({output_key[i]:tmp['output_data'][:, i] for i in range(len(output_key))})
            df = pd.DataFrame(excel_dict)
            df.to_excel(writer, sheet_name=name, index=False)
        writer.save()
        writer.close()
    
    
    