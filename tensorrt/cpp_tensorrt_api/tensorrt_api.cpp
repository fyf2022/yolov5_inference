#include<windows.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include <time.h>
#include "npp.h"
#include "npps_support_functions.h"

#include "box.hpp"

using namespace std;
using namespace cv;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

extern "C"
	void decode_kernel_invoker( float* predict, int num_bboxes, int num_classes, float confidence_threshold, float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)



// @brief 用于创建IBuilder、IRuntime或IRefitter实例的记录器用于通过该接口创建的所有对象。
// 在释放所有创建的对象之前，记录器应一直有效。
// 主要是实例化ILogger类下的log()方法。
class Logger : public nvinfer1::ILogger{
	void log(Severity severity, const char* message)  noexcept{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << message << std::endl;
	}
} gLogger;

// @brief 
typedef struct tensorRT_nvinfer {
	Logger logger;
	// 反序列化引擎
	nvinfer1::IRuntime* runtime;
	// 推理引擎
	// 保存模型的模型结构、模型参数以及最优计算kernel配置；
	// 不能跨平台和跨TensorRT版本移植
	nvinfer1::ICudaEngine* engine;
	// 上下文
	// 储存中间值，实际进行推理的对象
	// 由engine创建，可创建多个对象，进行多推理任务
	nvinfer1::IExecutionContext* context;
	// cudn缓存标志
	cudaStream_t stream;
	// GPU显存输入/输出缓冲
	void** data_buffer;
	int input_w;
	int input_h;
	int input_c;
	int feature_dim;
	int class_num;
} NvinferStruct;



// @brief 将wchar_t*字符串指针转换为string字符串格式
// @param wchar 输入字符指针
// @return 转换出的string字符串 
std::string wchar_to_string(const wchar_t* wchar) {
	// 获取输入指针的长度
	int path_size = WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
	char* chars = new char[path_size + 1];
	// 将双字节字符串转换成单字节字符串
	WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), chars, path_size, NULL, NULL);
	chars[path_size] = '\0';
	std::string pattern = chars;
	delete chars; //释放内存
	return pattern;
}

// @brief 将wchar_t*字符串指针转换为string字符串格式
// @param wchar 输入字符指针
// @return 转换出的string字符串 
char* wchar_to_char(const wchar_t* wchar) {
	// 获取输入指针的长度
	int path_size = WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
	char* chars = new char[path_size + 1];
	// 将双字节字符串转换成单字节字符串
	WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), chars, path_size, NULL, NULL);
	chars[path_size] = '\0';
	return chars;
}

// @brief 将图片的矩阵数据转换为opencv的mat数据
// @param data 图片矩阵
// @param size 图片矩阵长度
// @return 转换后的mat数据
cv::Mat data_to_mat(uchar* data, int image_Cols, int image_Rows) {
	cv::Mat m1(image_Rows, image_Cols, CV_8UC3, data);
	return m1;
}

// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
// @param onnx_file_path_wchar onnx模型本地地址
// @param engine_file_path_wchar engine模型本地地址
// @param type 输出模型精度，
extern "C"  __declspec(dllexport) void __stdcall onnx_to_engine(const wchar_t* onnx_file_path_wchar,
	const wchar_t* engine_file_path_wchar, int type) {
	std::string onnx_file_path = wchar_to_string(onnx_file_path_wchar);
	std::string engine_file_path = wchar_to_string(engine_file_path_wchar);

	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	// 定义网络属性
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
	// onnx文件解析类
	// 将onnx文件解析，并填充rensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	// 解析onnx文件
	parser->parseFromFile(onnx_file_path.c_str(), 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小。
	config->setMaxWorkspaceSize(16 * (1 << 20));
	// 设置模型输出精度
	if (type == 1) {
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	if (type == 2) {
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
	}
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理银枪保存到本地
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream file_ptr(engine_file_path, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* model_stream = engine->serialize();
	// 将文件保存到本地
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
	// 销毁创建的对象
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}


// @brief 读取本地engine模型，并初始化NvinferStruct
// @param engine_filename_wchar engine本地模型地址
// @param num_ionode 显存缓冲区数量
// @return NvinferStruct结构体指针
extern "C"  __declspec(dllexport) void* __stdcall nvinfer_init(const wchar_t* engine_filename_wchar, int num_ionode, const wchar_t*  input_node_name_wchar, const wchar_t*  output_node_name_wchar) {
	// 读取本地模型文件
	std::string engine_filename = wchar_to_string(engine_filename_wchar);
	const char* input_node_name = wchar_to_char(input_node_name_wchar);
	const char* output_node_name = wchar_to_char(output_node_name_wchar);
	// 以二进制方式读取问价
	std::ifstream file_ptr(engine_filename, std::ios::binary);
	if (!file_ptr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
	}

	size_t size = 0;
	file_ptr.seekg(0, file_ptr.end);	// 将读指针从文件末尾开始移动0个字节
	size = file_ptr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	file_ptr.seekg(0, file_ptr.beg);	// 将读指针从文件开头开始移动0个字节
	char* model_stream = new char[size];
	file_ptr.read(model_stream, size);
	// 关闭文件
	file_ptr.close();

	// 创建推理核心结构体，初始化变量
	NvinferStruct* p = new NvinferStruct();
	// 初始化反序列化引擎
	p->runtime = nvinfer1::createInferRuntime(gLogger);
	// 初始化推理引擎
	p->engine = p->runtime->deserializeCudaEngine(model_stream, size);
	// 创建上下文
	p->context = p->engine->createExecutionContext();
	// 创建gpu数据缓冲区
	p->data_buffer = new void*[num_ionode];

	int input_node_index = p->engine->getBindingIndex(input_node_name);
	nvinfer1::Dims input_node_dim = p->engine->getBindingDimensions(input_node_index);
	p->input_c = input_node_dim.d[1];
	p->input_w = input_node_dim.d[2];
	p->input_h = input_node_dim.d[3];

	int output_node_index = p->engine->getBindingIndex(output_node_name);
	nvinfer1::Dims output_node_dim = p->engine->getBindingDimensions(output_node_index);
	p->feature_dim = output_node_dim.d[1];
	p->class_num = output_node_dim.d[2];

	delete[] model_stream;
	return (void*)p;
}

// @brief 创建GPU显存输入/输出缓冲区
// @param nvinfer_ptr NvinferStruct结构体指针
// @para node_name_wchar 网络节点名称
// @param data_length 缓冲区数据长度
// @return NvinferStruct结构体指针
extern "C"  __declspec(dllexport) void* __stdcall creat_gpu_buffer(void* nvinfer_ptr,
	const wchar_t* node_name_wchar, size_t data_length) {
	// 重构NvinferStruct
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	const char* node_name = wchar_to_char(node_name_wchar);
	// 获取节点序号
	int node_index = p->engine->getBindingIndex(node_name);
	// 创建指定节点GPU显存缓冲区
	cudaMalloc(&(p->data_buffer[node_index]), data_length * sizeof(float));
	return (void*)p;
}

extern "C"  __declspec(dllexport) void __stdcall read_model_info(void* nvinfer_ptr, const wchar_t*  input_node_name_wchar, const wchar_t*  output_node_name_wchar, int* model_info) {
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	const char* input_node_name = wchar_to_char(input_node_name_wchar);
	const char* output_node_name = wchar_to_char(output_node_name_wchar);
	int input_node_index = p->engine->getBindingIndex(input_node_name);
	nvinfer1::Dims input_node_dim = p->engine->getBindingDimensions(input_node_index);

	int output_node_index = p->engine->getBindingIndex(output_node_name);
	nvinfer1::Dims output_node_dim = p->engine->getBindingDimensions(output_node_index);

	*model_info = input_node_dim.d[1];;
	model_info++;
	*model_info = input_node_dim.d[2];;
	model_info++;
	*model_info = input_node_dim.d[3];;
	model_info++;
	*model_info = output_node_dim.d[1];
	model_info++;
	*model_info = output_node_dim.d[2];
	model_info++;
}

// @brief 加载图片输入数据到缓冲区
// @param nvinfer_ptr NvinferStruct结构体指针
// @para node_name_wchar 网络节点名称
// @param image_data 图片矩阵数据
// @param image_size 图片数据长度
// @return NvinferStruct结构体指针
extern "C"  __declspec(dllexport) void* __stdcall load_image_data(void* nvinfer_ptr,
	const wchar_t* node_name_wchar, uchar * image_data, int img_w, int img_h, int BN_means) {
	// 重构NvinferStruct
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;

	// 获取输入节点信息
	const char* node_name = wchar_to_char(node_name_wchar);
	int node_index = p->engine->getBindingIndex(node_name);
	// 获取输入节点未读信息
	int INPUT_W = p->input_w;
	int INPUT_H = p->input_h;
	int INPUT_C = p->input_c;
	// 输入节点二维形状
	cv::Size node_shape(INPUT_W, INPUT_H);
	// 输入节点二维大小
	size_t node_data_length = INPUT_W * INPUT_H * INPUT_C;

	// 预处理输入数据
	cv::Mat input_image = data_to_mat(image_data, img_w, img_h);
	uchar* img_data = input_image.data;
	int img_c = input_image.channels();

	if (BN_means == 0)
	{
		std::vector<float> input_data(node_data_length);
		// 将图像归一化，并放缩到指定大小
		input_image = cv::dnn::blobFromImage(input_image, 1 / 255.0, node_shape, cv::Scalar(0, 0, 0), true, false);
		// 将图片数据copy到输入流中
		memcpy(input_data.data(), input_image.ptr<float>(), node_data_length * sizeof(float));

		// 创建cuda流
		cudaStreamCreate(&p->stream);

		// 将输入数据由内存到GPU显存
		cudaMemcpyAsync(p->data_buffer[node_index], input_data.data(), node_data_length * sizeof(float), cudaMemcpyHostToDevice, p->stream);

		std::vector<float>().swap(input_data);
		input_image.release();
	}
	else if (BN_means == 1)
	{
		std::vector<float> input_data(node_data_length);
		cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB); // 将图片通道由 BGR 转为 RGB
		// 对输入图片按照tensor输入要求进行缩放
		cv::resize(input_image, input_image, node_shape, 0, 0, cv::INTER_CUBIC);
		// 图像数据归一化
		input_image.convertTo(input_image, CV_32FC3, 1.0f / 255.0);
		std::vector<cv::Mat> input_channels(3);
		cv::split(input_image, input_channels);
		//std::vector<float> input_data(node_dim.d[2] * node_dim.d[3] * 3);
		auto data = input_data.data();
		int channelLength = INPUT_W * INPUT_H;
		for (int i = 0; i < 3; ++i)
		{
			memcpy(data, input_channels[i].data, channelLength * sizeof(float));
			data += channelLength;
		}
		// 创建cuda流
		cudaStreamCreate(&p->stream);

		// 将输入数据由内存到GPU显存
		cudaMemcpyAsync(p->data_buffer[node_index], input_data.data(), node_data_length * sizeof(float), cudaMemcpyHostToDevice, p->stream);

		std::vector<float>().swap(input_data);
		input_image.release();
	}
	else
	{
		Npp8u *gpu_img_resize_buf;
		Npp32f *gpu_data_buf;
		Npp32f *gpu_data_planes;
		NppiSize dstSize = { INPUT_W, INPUT_H };;
		NppiRect dstROI = { 0, 0, INPUT_W, INPUT_H };;
		Npp32f m_scale[3] = { 0.00392157, 0.00392157, 0.00392157 };
		int aDstOrder[3] = { 2, 1, 0 };
		CHECK(cudaMalloc(&gpu_img_resize_buf, INPUT_W * INPUT_H * INPUT_C * sizeof(uchar)));
		CHECK(cudaMalloc(&gpu_data_buf, INPUT_W * INPUT_H * INPUT_C * sizeof(float)));
		CHECK(cudaMalloc(&gpu_data_planes, INPUT_W * INPUT_H * INPUT_C * sizeof(float)));

		NppiSize srcSize = { img_w, img_h };
		NppiRect srcROI = { 0, 0, img_w, img_h };
		Npp8u *gpu_img_buf;
		CHECK(cudaMalloc(&gpu_img_buf, img_w * img_h * img_c * sizeof(uchar)));
		Npp32f* dst_planes[3] = { gpu_data_planes, gpu_data_planes + INPUT_W * INPUT_H, gpu_data_planes + INPUT_W * INPUT_H * 2 };

		cudaMemcpy(gpu_img_buf, img_data, img_w*img_h * img_c, cudaMemcpyHostToDevice);
		nppiResize_8u_C3R(gpu_img_buf, img_w * img_c, srcSize, srcROI, gpu_img_resize_buf, INPUT_W * INPUT_C, dstSize, dstROI, NPPI_INTER_LINEAR);
		nppiSwapChannels_8u_C3IR(gpu_img_resize_buf, INPUT_W * INPUT_C, dstSize, aDstOrder);
		nppiConvert_8u32f_C3R(gpu_img_resize_buf, INPUT_W * INPUT_C, gpu_data_buf, INPUT_W * INPUT_C * sizeof(float), dstSize);
		nppiMulC_32f_C3IR(m_scale, gpu_data_buf, INPUT_W * INPUT_C * sizeof(float), dstSize);
		nppiCopy_32f_C3P3R(gpu_data_buf, INPUT_W * INPUT_C * sizeof(float), dst_planes, INPUT_W * sizeof(float), dstSize);

		cudaStreamCreate(&p->stream);
		// 将输入数据由内存到GPU显存
		cudaMemcpyAsync(p->data_buffer[node_index], gpu_data_planes, INPUT_C * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyDeviceToDevice, p->stream);
		CHECK(cudaFree(gpu_img_buf));
		CHECK(cudaFree(gpu_img_resize_buf));
		CHECK(cudaFree(gpu_data_buf));
		CHECK(cudaFree(gpu_data_planes));
	}
	return (void*)p;

}


// @brief 模型推理
// @param nvinfer_ptr NvinferStruct结构体指针
// @return NvinferStruct结构体指针
extern "C"  __declspec(dllexport) void* __stdcall infer(void* nvinfer_ptr) {
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	// 模型推理
	p->context->enqueueV2(p->data_buffer, p->stream, nullptr);
	return (void*)p;
}


// @brief 读取推理数据
// @param nvinfer_ptr NvinferStruct结构体指针
// @para node_name_wchar 网络节点名称
// @param output_result 输出数据指针
extern "C"  __declspec(dllexport) void __stdcall read_infer_result(void* nvinfer_ptr,
	const wchar_t* node_name_wchar, int* output_result, int NMS_means) {
	
	// 重构NvinferStruct
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	int feature_dim = p->feature_dim;
	int class_num = p->class_num;
	size_t node_data_length = feature_dim * class_num;

	// 获取输出节点信息
	const char* node_name = wchar_to_char(node_name_wchar);
	int node_index = p->engine->getBindingIndex(node_name);

	if (NMS_means==0)
	{
		// 读取输出数据
		// 创建输出数据
		float* result = new float[node_data_length];
		// 将输出数据由GPU显存到内存
		const clock_t t0 = clock();
		cudaMemcpyAsync(result, p->data_buffer[node_index], node_data_length * sizeof(float), cudaMemcpyDeviceToHost, p->stream);

		cv::Mat det_output = cv::Mat(feature_dim, class_num, CV_32F, result);
		//// post-process
		std::vector<cv::Rect> position_boxes;
		std::vector<int> classIds;
		std::vector<float> confidences;
		for (int i = 0; i < det_output.rows; i++) {
			float confidence = det_output.at<float>(i, 4);
			if (confidence < 0.2) {
				continue;
			}
			cv::Mat classes_scores = det_output.row(i).colRange(5, class_num);
			cv::Point classIdPoint;
			double score;
			// 获取一组数据中最大值及其位置
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
			// 置信度 0～1之间
			if (score > 0.25)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				int x = static_cast<int>(cx - 0.5 * ow);
				int y = static_cast<int>(cy - 0.5 * oh);
				int width = static_cast<int>(ow);
				int height = static_cast<int>(oh);
				cv::Rect box;
				box.x = x;
				box.y = y;
				box.width = width;
				box.height = height;

				position_boxes.push_back(box);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}
		// NMS
		std::vector<int> indexes;
		cv::dnn::NMSBoxes(position_boxes, confidences, 0.25, 0.45, indexes);

		for (size_t i = 0; i < indexes.size(); i++) {
			int index = indexes[i];
			*output_result = classIds[index];
			output_result++;
			*output_result = confidences[index] * 1000;
			output_result++;
			*output_result = position_boxes[index].tl().x;
			output_result++;
			*output_result = position_boxes[index].tl().y;
			output_result++;
			*output_result = position_boxes[index].width;
			output_result++;
			*output_result = position_boxes[index].height;
			output_result++;
		}
		delete[] result;
		std::vector<cv::Rect>().swap(position_boxes);
		std::vector<int>().swap(classIds);
		std::vector<float>().swap(confidences);
	}
	else
	{
		float confidence_threshold = 0.25f; 
		float nms_threshold = 0.45f;
		vector<Box> box_result;
		cudaStream_t stream = nullptr;
		checkRuntime(cudaStreamCreate(&stream));

		float* output_device = nullptr;
		float* output_host = nullptr;
		int max_objects = 1000;
		int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
		checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
		//output_host = (float*)malloc(sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float));
		//memset(output_host, 0, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float));
		checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));

		//checkRuntime(cudaStreamSynchronize(stream));
		cudaDeviceSynchronize();
		
		decode_kernel_invoker(
			(float *)p->data_buffer[node_index], feature_dim, class_num - 5, confidence_threshold,
			nms_threshold, nullptr, output_device, max_objects, NUM_BOX_ELEMENT, stream
		);
		
		
		checkRuntime(cudaMemcpy(output_host, output_device,
			sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float),
			cudaMemcpyDeviceToHost));
		
		int num_boxes = min((int)output_host[0], max_objects);
		for (int i = 0; i < num_boxes; ++i) {
			float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
			int keep_flag = ptr[6];
			if (keep_flag) {
				*output_result = (int)ptr[5];
				output_result++;
				*output_result = (int)(ptr[4] * 1000);
				output_result++;
				*output_result = (int)ptr[0];
				output_result++;
				*output_result = (int)ptr[1];
				output_result++;
				*output_result = (int)(ptr[2] - ptr[0]);
				output_result++;
				*output_result = (int)(ptr[3] - ptr[1]);
				output_result++;
				// left, top, right, bottom, confidence, class, keepflag
			}
		}
		
		/*
		checkRuntime(cudaStreamDestroy(stream));
		//checkRuntime(cudaFree(output_device));
		//free(output_host);
		checkRuntime(cudaFreeHost(output_host));
		*/
	}
}

// @brief 删除内存地址
// @param nvinfer_ptr NvinferStruct结构体指针
extern "C"  __declspec(dllexport) void __stdcall nvinfer_delete(void* nvinfer_ptr) {
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	delete p->data_buffer;
	p->context->destroy();
	p->engine->destroy();
	p->runtime->destroy();
	delete p;
}