# Read

***本文件是赛题7的验收使用说明，如果遇到验收问题，请及时与我们沟通，感谢！！！***

## 文件说明

我们的工程文件总代码全部在**home**目录下的**pjr**文件夹下(**/home/eda210707/prj**)。

进入文件夹，有如下几个文件夹

![image-20211212111807933](C:\Users\尹硕\AppData\Roaming\Typora\typora-user-images\image-20211212111807933.png)

其中**casex**对应第**x**道题目的代码

以**case4**的运行为例，相关程序文件在/home/eda210707/prj/case4文件夹中。

打开相应的目录，初始情况下，我们有如下的文件（以**case4**为例）：

![image-20211212152711734](C:\Users\尹硕\AppData\Roaming\Typora\typora-user-images\image-20211212152711734.png)

其中**BO_System.py**，**GP_System.py**为我们的贝叶斯优化模型代码和高斯过程模型代码，**lib4.cc**为C++顶层代码，**lib4.so**是g++编译后的共享库文件，**edaInterface.h**是C++接口说明。

## 使用说明

+ 第一步，进入**case1**目录：首先使用命令：

~~~~shell
cd /home/eda210707/prj/case1
~~~~

进入**case1**文件夹，使用**ls**查看目录，包括如图的文件

![image-20211212143458466](C:\Users\尹硕\AppData\Roaming\Typora\typora-user-images\image-20211212143458466.png)



+ 第二步，编译**C++**文件：使用命令：

~~~~shell
g++ -fPIC -shared lib1.cc -o lib1.so 
~~~~

将文件**lib1.cc**编译为**lib1.so**

| 5个case的编译命令 |                                       |
| ----------------- | ------------------------------------- |
| Case1             | g++ -fPIC  -shared lib1.cc -o lib1.so |
| Case2             | g++ -fPIC  -shared lib2.cc -o lib2.so |
| Case3             | g++ -fPIC  -shared lib3.cc -o lib3.so |
| Case4             | g++ -fPIC  -shared lib4.cc -o lib4.so |
| Case5             | g++ -fPIC  -shared lib5.cc -o lib5.so |

**（注意一定要到对应的文件目录【参考第一步】）*  



+ 第三步，运行程序：使用命令：

~~~~shell
nanospice /export/testcases/case1/edaCase1.sp +edaLib=/home/eda210707/prj/case1/lib1.so -mp 8
~~~~

| 5个case的运行命令（注意一定要到对应的文件目录【参考第一步】） |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Case1                                                        | nanospice  /export/testcases/case1/edaCase1.sp +edaLib=/home/eda210707/prj/case1/lib1.so  -mp 8 |
| Case2                                                        | nanospice  /export/testcases/case2/edaDemoCase.sp +edaLib=/home/eda210707/prj/case2/lib2.so  -mp 8 |
| Case3                                                        | nanospice  /export/testcases/case3/case3.sp +edaLib=/home/eda210707/prj/case3/lib3.so  -mp 8 |
| Case4                                                        | nanospice  /export/testcases/case4/case4.sp +edaLib=/home/eda210707/prj/case4/lib4.so  -mp 8 |
| Case5                                                        | nanospice  /export/testcases/case5/case5.sp +edaLib=/home/eda210707/prj/case5/lib5.so  -mp 8 |

出现**“result saved”**输出表示仿真和运算结束，最终结果保存在**/home/eda210707/prj/case1/edaCase1.output/result.csv**

| 5个case结果的保存位置（当前目录为/home/eda210707/prj） |                                     |
| ------------------------------------------------------ | ----------------------------------- |
| Case1                                                  | case1/edaCase1.output/result.csv    |
| Case2                                                  | case2/edaDemoCase.output/result.csv |
| Case3                                                  | case3/case3.output/result.csv       |
| Case4                                                  | case4/case4.output/result.csv       |
| Case5                                                  | case5/case5.output/result.csv       |

$$
格式:=序号+结果+输入
$$



## Requires

+ pytorch 1.9.0
+ gpytorch  1.5.0
+ botorch 0.5.1

+ numpy 1.20.3
+ pandas 1.3.2