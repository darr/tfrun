#### 运行方式 在sh_file目录下的 run_app.sh
---
>运行 sh run_app.sh 
``` shell
sh run_app.sh
```

1. [ ] 运行失败的话，尝试运行 sudo －H sh run_app.sh  
run_app.sh shell 会创建python虚拟运行环境，当然这要求环境中有virtualenv
然后自动下载依赖的模块 numpy matplotlib karas 最重要的tensorflow 有时会出现现在超时，换源或者把超时时间设长，总之自己想办法吧。
1. [ ] 第一次创建失败的话，活着必备模块没有下载成功，可以一直运行sh_file目录下的create_env.sh

``` shell
sh create_env.sh
```
或者

``` shell
sudo －H sh create_env.sh
```


#### 关于文件

1. 代码结构

    | 文件加       | 描述 |作用|
    | :-------- | -----    |----|
    | pybase      | python 基础工具包|可服用的python模块，包括日志网络等|
    | sh_file      | shell 文件     |执行一些批处理，设置环境|
    | etc.py      | 程序的默认配置文件    |可以在这里添加默认配置|
    | appargs.py      | 程序的可传参数文件    |可以增加减少参数|
    | mnist.py      | 程序的入口文件    |包含各种处理mnist数据集的方法|
    | dealMnist.py      | 尝试将tf的图分开    |效果不好，跟想象不一样，不过理解ok|
    | runMnist.py      | 没有分开图    |不必过度抽象，有slim和kares 理解就好|



