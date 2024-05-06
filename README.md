#程序功能
用于gaussion splatting 生成
1) 读取fbx 生成多视角视图，并生成transforms_train.json和points3d.ply 文件
2) 读取一张纹理生成无光照材质，付给主模型！！！


#使用方法
需要 blender 作为外部程序启动
已经配置了启动任务，在 tasks.json 中进行了配置

运行 ctrl+shift+b 启动程序


#注意事项
1) 输入的fbx文件需要预先处理。只能有一个mesh，名字定为‘main’  放在data目录中
2) 需要有一个main.png作为输入的纹理
3) 需要将blender 可执行程序加到windwos系统的path中！！！

#输入要求
输入的模型:data/main.fbx
输入的纹理:dtata/main.png
