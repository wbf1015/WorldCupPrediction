来源于公众号datayx

但其中部分库函数调用的函数已经是老版本中的函数，如Tensorflow中的predict_class 如果是最近更新的包可能会发现跑不通程序的现象

同样，通过隐藏报错防止弹出CPU和TensorFlow“不兼容”的问题，也就是说也许你的 CPU 支持AVX AVX2 （可以加速CPU计算），但你安装的 TensorFlow 版本不支持

对上面涉及的问题做了简单的修改，仅供娱乐~
