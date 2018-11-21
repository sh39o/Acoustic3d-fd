# Acoustic3d
三维声波方程正演程序。（二阶声波方程，一阶交错网格声波方程，一阶vti交错网格声波方程）

支持二阶声波方程，一阶各向同性声波方程和一阶vti各向异性拟声波方程，基于Rong Tao的代码。

支持多线程计算，同时将任务分配给多个GPU

支持输出波长快照

支持输出炮记录

支持切除直达波

要求python3

安装和运行：
make
在pyAcoustic.py设置参数
其中nnode为一共的节点数，nodei为当前节点数，单机就将nnode写为1
设置正演算子（二阶，一阶，一阶vti）

二阶声波方程，Acoustic3d2order，只需要速度模型文件

一阶声波方程，Acoustic3d1order，需要速度和密度模型文件

一阶vti声波方程，Acoustic3dvti，需要速度，密度，delta，epsilon文件

执行python3 ./pyAcoustic.py

pyAcoustic.py为主程序，readmodel为配置的类文件，src目录下为cuda代码

