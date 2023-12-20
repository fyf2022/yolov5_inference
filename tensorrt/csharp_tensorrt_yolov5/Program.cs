using System;
using OpenCvSharp;
using TensorRtSharp;
using OpenCvSharp.Dnn;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;



namespace charp_tensorrt_yolov5
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 模型基本信息
            //string model_path_onnx = "D:/vs_code/Inference-master/model/yolov5/yolov5s.onnx";
            string engine_path = "D:/vs_code/Inference-master/model/yolov5/yolov5s.engine";
            string images_path = "D:/vs_code/Inference-master/model/yolov5/test_image";
            string lable_path = "D:/vs_code/Inference-master/model/yolov5/lable.txt";
            string input_node_name = "images";
            string output_node_name = "output";

            //模型转换
            Nvinfer nvinfer= new Nvinfer();
            //nvinfer.onnx_to_engine(model_path_onnx, engine_path, AccuracyFlag.kFP16);

            // 创建模型推理类
            System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();
            System.Diagnostics.Stopwatch watch2 = new System.Diagnostics.Stopwatch();

            watch.Start();
            //Nvinfer nvinfer = new Nvinfer();
            // 读取模型信息
            nvinfer.init(engine_path, 2, input_node_name, output_node_name);

            int[] model_info = nvinfer.read_model_info(input_node_name, output_node_name);
            ulong input_c = Convert.ToUInt64(model_info[0]);
            ulong input_w = Convert.ToUInt64(model_info[1]);
            ulong input_h = Convert.ToUInt64(model_info[2]);
            ulong feat_dim = Convert.ToUInt64(model_info[3]);
            //默认类别数加5
            ulong class_num = Convert.ToUInt64(model_info[4]);

            // 配置输入输出gpu缓存区
            nvinfer.creat_gpu_buffer(input_node_name, input_c * input_h * input_w);
            nvinfer.creat_gpu_buffer(output_node_name, feat_dim * class_num);

            watch.Stop();
            TimeSpan timeSpan0 = watch.Elapsed;
            System.Console.WriteLine("model load time: {0}(ms)", timeSpan0.TotalMilliseconds);
            watch.Restart();
            

            var files = Directory.GetFiles(images_path, "*.jpg");
            foreach (string img_path in files)
            {
                Console.WriteLine("*********************************************");
                string img_name = Path.GetFileName(img_path); 
                Console.WriteLine(img_name);
                // 配置图片数据
                watch2.Restart();
                watch2.Start();
                Mat image = new Mat(img_path);
                
                watch2.Stop();
                TimeSpan timeSpan5_22 = watch2.Elapsed;
                System.Console.WriteLine("图片读取: {0}(ms)", timeSpan5_22.TotalMilliseconds);
                watch2.Restart();

                int image_Cols = image.Cols;
                int image_Rows = image.Rows;

                //int max_image_length = image.Cols > image.Rows ? image.Cols : image.Rows;
                var image_data1 = new byte[image.Total() * 3];//这里必须乘以通道数，不然数组越界，也可以用w*h*c，差不多
                                                              //image.GetArray(out image_data1);
                Marshal.Copy(image.Data, image_data1, 0, image_data1.Length);
                watch2.Stop();
                TimeSpan timeSpan5_1 = watch2.Elapsed;
                System.Console.WriteLine("C#传递图片到数组 time1: {0}(ms)", timeSpan5_1.TotalMilliseconds);
                watch2.Restart();

                watch.Restart();
                watch.Start();
                // 加载推理图片数据
                nvinfer.load_image_data(input_node_name, image_data1, image_Cols, image_Rows, BNFlag.fyf_cpu);
                watch.Stop();
                TimeSpan timeSpan1 = watch.Elapsed;
                System.Console.WriteLine("加载推理图片time: {0}(ms)", timeSpan1.TotalMilliseconds);
                watch.Restart();

                // 模型推理
                watch.Start();
                nvinfer.infer();
                watch.Stop();
                TimeSpan timeSpan2 = watch.Elapsed;
                System.Console.WriteLine("inference time: {0}(ms)", timeSpan2.TotalMilliseconds);
                watch.Restart();

                // 读取推理结果
                watch.Start();
                int[] result_array = nvinfer.read_infer_result(output_node_name, MMSFlag.fyf_gpu);
                watch.Stop();
                TimeSpan timeSpan21 = watch.Elapsed;
                System.Console.WriteLine("nvinfer.read_infer_result time: {0}(ms)", timeSpan21.TotalMilliseconds);
                watch.Restart();

                
                float factor_x = image_Cols / (float)input_w;
                float factor_y = image_Rows / (float)input_h;
                for (int i = 0; i < result_array.Length / 6; i++)
                {
                    int index = i * 6;
                    int score = result_array[index + 1];
                    //System.Console.WriteLine("score: {0}", score);
                    if (score == 0)
                    {
                        break;
                    }
                    else
                    {
                        int class_idx = result_array[index];
                        Rect box = new Rect();
                        box.X = (int)(result_array[index + 2] * factor_x);
                        box.Y = (int)(result_array[index + 3] * factor_y);
                        box.Width = (int)(result_array[index + 4] * factor_x);
                        box.Height = (int)(result_array[index + 5] * factor_y);
                        Cv2.Rectangle(image, box, new Scalar(0, 0, 255), 5, LineTypes.Link8);
                        Cv2.Rectangle(image, new Point(box.TopLeft.X, box.TopLeft.Y - 20),
                            new Point(box.BottomRight.X, box.TopLeft.Y), new Scalar(0, 255, 255), -1);
                        Cv2.PutText(image, class_idx + "-" + (score / 1000.0).ToString("0.00"),
                            new Point(box.X, box.Y - 10),
                            HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 0, 0), 1);

                    }

                }
                //Array.Clear(result_array, 0, result_array.Length);
                Cv2.ImWrite("D:/process_imgs/" + img_name, image);
            }
            nvinfer.delete();
            Console.ReadKey();
        }
    }
}
