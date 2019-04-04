# alpha_sigma
目前可视化文件主要写在GUI.py,如果只想体验下游戏模型或者查看模型自我下棋时的对弈过程，只需要调用GUI.py就够了。

main.py 主程序，程序入口

GUI.py 用于交互的可视化界面

    我们在这里提供了两种模式：
    
      游戏模式： 从终端调用命令： python GUI.py --mode game --game_model model_5400.pkl  其中model_5400.pkl是已经训练好的神经网络，通过--mode指定模式为游戏模式，并通过--game_model装载已经训练好的模型。我们这里提供了一个模型文件：model_5400.pkl.(PS:家用机计算速度慢，下一步棋大概需要等7秒钟)
      
      展示模式： 展示神经网络训练过程中机器自我对弈的结果。从终端调用命令： python GUI.py --mode display --display_file test_5200.pkl 我们这里提供了游戏记录。
    

new_MCTS.py 蒙特卡罗树程序

network.py 神经网络程序

five_stone_game.py 五子棋游戏程序

utils.py 用来装闲杂文件

现在该套程序首发在知乎，知乎链接：https://zhuanlan.zhihu.com/p/59567014 欢迎大家去帮我点赞

English Version:
    
