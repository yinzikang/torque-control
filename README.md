# zsn
- func.py
包含了一些计算机器人属性的函数
- main.py、main_ccr.py
末端加有外力的阻抗控制
- ipd_K20_D100.csv、ipd_K40_D100.csv、ipd_K100_D100.csv
阻抗控制记录的数据
- traj.py
roboticstoolbox的python版本demo，可忽略
- jk5.xml
无外设的纯机械臂

# yinzikang
基于师兄阻抗控制的扩展，可以摆脱师兄的文件
基于jk5复现Force tracking impedance control of robot manipulators under unknown environment中力控制算法
- paper
整理后的可以多种控制方法运行的阻抗控制
- backup中paper_work, paper_work_1, paper_work_2均为未整理的老控制律，只为副本存在，可忽略
- funcmini.py
针对有连杆机器人的kdl