# YOLO v5 前后处理GPU加速部署

## 简介

​	深度学习模型部署中复杂的前后处理往往部署cpu中，这一方面极大地增加模型处理耗时，另一方面挤兑了cpu计算资源，导致整个软件平台卡死，本博客针对C#编写界面软件调用C++模型接口场景，分别从C#与C++间图片传输优化、前处理GPU加速、后处理加速三个角度进行优化。


## 项目文档

想要更详细的了解该项目，请参考博客[https://blog.csdn.net/ffyunfeng16/article/details/135042575](https://blog.csdn.net/ffyunfeng16/article/details/135042575)

## 使用环境

**系统平台：**

​			Windows

**软件要求（其他版本也可以支持需要更改一下路径）：**

​			Visual Studio 2017 

​			OpenCV 4.4.0（D:\Program Files\opencv\opencv_440）

​			CUDA 11.7（C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7）

​			CUDNN 8.7.0（cudnn-windows-x86_64-8.7.0.84_cuda11-archive.zip）

​			TensorRT 8.6.1.6（D:\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6\）

## 下载

**在Github上克隆下载：**

```shell
git clone https://github.com/fyf2022/yolov5_inference.git
```

**在Gitee上克隆下载：**

```shell
git clone https://gitee.com/fyf2022/yolov5_inference.git
```

## 项目说明
​	该项目利用数组将C#读取的图片传递给C++，避免编解码方式优化传递速度
```shell
Mat image = new Mat(img_path);
int image_Cols = image.Cols;
int image_Rows = image.Rows;
var image_data = new byte[image.Total() * 3];//这里必须乘以通道数，不然数组越界，也可以用w*h*c
Marshal.Copy(image.Data, image_data1, 0, image_data.Length);
```

​	该项目提供了提供了BNFlag.Normal来调用cv::dnn::blobFromImage的cpu前处理操作，BNFlag.fyf_cpu来调用自己实现的cpu版本C++前处理，MMSFlag.fyf_gpu来调用gpu版本前处理
```shell
nvinfer.load_image_data(input_node_name, image_data, image_Cols, image_Rows, BNFlag.Normal);
nvinfer.load_image_data(input_node_name, image_data, image_Cols, image_Rows, BNFlag.fyf_cpu);
nvinfer.load_image_data(input_node_name, image_data, image_Cols, image_Rows, BNFlag.fyf_gpu);
```
​	后处理包含NMS该项目提供了MMSFlag.fyf_cpu来调用cpu版本C++后处理，MMSFlag.fyf_gpu来调用gpu版本后处理
```shell
int[] result_array = nvinfer.read_infer_result(output_node_name, MMSFlag.fyf_cpu);
int[] result_array = nvinfer.read_infer_result(output_node_name, MMSFlag.fyf_gpu);

```

## 项目文件夹说明
​	项目启动路径（D:/vs_code/Inference-master/Inference.sln）

​	模型调用的C++接口（D:\vs_code\Inference-master\tensorrt\cpp_tensorrt_api）

​	C#调用C++接口的类（D:\vs_code\Inference-master\tensorrt\csharp_tensorrt_class）

​	C#主执行文件（D:\vs_code\Inference-master\tensorrt\csharp_tensorrt_yolov5）


