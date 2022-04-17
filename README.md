## 写在前面
这个项目随便写写，效果啥的图一乐，只能说这个数据集有毒，其中有一个`rand_select `模型就是纯粹用来娱乐的[doge]

## 实验效果
### MSE度量结果

- Zigbee

<img src="result/image-20220417221012510.png" alt="image-20220417221012510" style="zoom:50%;" />


- BLE

<img src="result/image-20220417221036565.png" alt="image-20220417221036565" style="zoom:50%;" />

- WiFi

<img src="result/image-20220417221059197.png" alt="image-20220417221059197" style="zoom:50%;" />

### CFD度量结果

直接以RSSI为输入

<img src="result/RSSI.jpg" alt="RSSI" style="zoom:50%;" />

用路径损失模型将RSSI转换为距离，作为输入

<img src="result/PATH.jpg" alt="PATH" style="zoom:50%;" />
