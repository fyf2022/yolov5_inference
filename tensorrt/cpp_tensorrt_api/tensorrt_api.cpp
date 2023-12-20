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



// @brief ���ڴ���IBuilder��IRuntime��IRefitterʵ���ļ�¼������ͨ���ýӿڴ��������ж���
// ���ͷ����д����Ķ���֮ǰ����¼��Ӧһֱ��Ч��
// ��Ҫ��ʵ����ILogger���µ�log()������
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
	// �����л�����
	nvinfer1::IRuntime* runtime;
	// ��������
	// ����ģ�͵�ģ�ͽṹ��ģ�Ͳ����Լ����ż���kernel���ã�
	// ���ܿ�ƽ̨�Ϳ�TensorRT�汾��ֲ
	nvinfer1::ICudaEngine* engine;
	// ������
	// �����м�ֵ��ʵ�ʽ�������Ķ���
	// ��engine�������ɴ���������󣬽��ж���������
	nvinfer1::IExecutionContext* context;
	// cudn�����־
	cudaStream_t stream;
	// GPU�Դ�����/�������
	void** data_buffer;
	int input_w;
	int input_h;
	int input_c;
	int feature_dim;
	int class_num;
} NvinferStruct;



// @brief ��wchar_t*�ַ���ָ��ת��Ϊstring�ַ�����ʽ
// @param wchar �����ַ�ָ��
// @return ת������string�ַ��� 
std::string wchar_to_string(const wchar_t* wchar) {
	// ��ȡ����ָ��ĳ���
	int path_size = WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
	char* chars = new char[path_size + 1];
	// ��˫�ֽ��ַ���ת���ɵ��ֽ��ַ���
	WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), chars, path_size, NULL, NULL);
	chars[path_size] = '\0';
	std::string pattern = chars;
	delete chars; //�ͷ��ڴ�
	return pattern;
}

// @brief ��wchar_t*�ַ���ָ��ת��Ϊstring�ַ�����ʽ
// @param wchar �����ַ�ָ��
// @return ת������string�ַ��� 
char* wchar_to_char(const wchar_t* wchar) {
	// ��ȡ����ָ��ĳ���
	int path_size = WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
	char* chars = new char[path_size + 1];
	// ��˫�ֽ��ַ���ת���ɵ��ֽ��ַ���
	WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), chars, path_size, NULL, NULL);
	chars[path_size] = '\0';
	return chars;
}

// @brief ��ͼƬ�ľ�������ת��Ϊopencv��mat����
// @param data ͼƬ����
// @param size ͼƬ���󳤶�
// @return ת�����mat����
cv::Mat data_to_mat(uchar* data, int image_Cols, int image_Rows) {
	cv::Mat m1(image_Rows, image_Cols, CV_8UC3, data);
	return m1;
}

// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
// @param onnx_file_path_wchar onnxģ�ͱ��ص�ַ
// @param engine_file_path_wchar engineģ�ͱ��ص�ַ
// @param type ���ģ�;��ȣ�
extern "C"  __declspec(dllexport) void __stdcall onnx_to_engine(const wchar_t* onnx_file_path_wchar,
	const wchar_t* engine_file_path_wchar, int type) {
	std::string onnx_file_path = wchar_to_string(onnx_file_path_wchar);
	std::string engine_file_path = wchar_to_string(engine_file_path_wchar);

	// ����������ȡcuda�ں�Ŀ¼�Ի�ȡ����ʵ��
	// ���ڴ���config��network��engine����������ĺ�����
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	// ������������
	const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// ����onnx�����ļ�
	// tensorRTģ����
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
	// onnx�ļ�������
	// ��onnx�ļ������������rensorRT����ṹ
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	// ����onnx�ļ�
	parser->parseFromFile(onnx_file_path.c_str(), 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// ������������
	// �������������ö���
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// ����������ռ��С��
	config->setMaxWorkspaceSize(16 * (1 << 20));
	// ����ģ���������
	if (type == 1) {
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	if (type == 2) {
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
	}
	// ������������
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// ��������ǹ���浽����
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream file_ptr(engine_file_path, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	// ��ģ��ת��Ϊ�ļ�������
	nvinfer1::IHostMemory* model_stream = engine->serialize();
	// ���ļ����浽����
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
	// ���ٴ����Ķ���
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}


// @brief ��ȡ����engineģ�ͣ�����ʼ��NvinferStruct
// @param engine_filename_wchar engine����ģ�͵�ַ
// @param num_ionode �Դ滺��������
// @return NvinferStruct�ṹ��ָ��
extern "C"  __declspec(dllexport) void* __stdcall nvinfer_init(const wchar_t* engine_filename_wchar, int num_ionode, const wchar_t*  input_node_name_wchar, const wchar_t*  output_node_name_wchar) {
	// ��ȡ����ģ���ļ�
	std::string engine_filename = wchar_to_string(engine_filename_wchar);
	const char* input_node_name = wchar_to_char(input_node_name_wchar);
	const char* output_node_name = wchar_to_char(output_node_name_wchar);
	// �Զ����Ʒ�ʽ��ȡ�ʼ�
	std::ifstream file_ptr(engine_filename, std::ios::binary);
	if (!file_ptr.good()) {
		std::cerr << "�ļ��޷��򿪣���ȷ���ļ��Ƿ���ã�" << std::endl;
	}

	size_t size = 0;
	file_ptr.seekg(0, file_ptr.end);	// ����ָ����ļ�ĩβ��ʼ�ƶ�0���ֽ�
	size = file_ptr.tellg();	// ���ض�ָ���λ�ã���ʱ��ָ���λ�þ����ļ����ֽ���
	file_ptr.seekg(0, file_ptr.beg);	// ����ָ����ļ���ͷ��ʼ�ƶ�0���ֽ�
	char* model_stream = new char[size];
	file_ptr.read(model_stream, size);
	// �ر��ļ�
	file_ptr.close();

	// ����������Ľṹ�壬��ʼ������
	NvinferStruct* p = new NvinferStruct();
	// ��ʼ�������л�����
	p->runtime = nvinfer1::createInferRuntime(gLogger);
	// ��ʼ����������
	p->engine = p->runtime->deserializeCudaEngine(model_stream, size);
	// ����������
	p->context = p->engine->createExecutionContext();
	// ����gpu���ݻ�����
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

// @brief ����GPU�Դ�����/���������
// @param nvinfer_ptr NvinferStruct�ṹ��ָ��
// @para node_name_wchar ����ڵ�����
// @param data_length ���������ݳ���
// @return NvinferStruct�ṹ��ָ��
extern "C"  __declspec(dllexport) void* __stdcall creat_gpu_buffer(void* nvinfer_ptr,
	const wchar_t* node_name_wchar, size_t data_length) {
	// �ع�NvinferStruct
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	const char* node_name = wchar_to_char(node_name_wchar);
	// ��ȡ�ڵ����
	int node_index = p->engine->getBindingIndex(node_name);
	// ����ָ���ڵ�GPU�Դ滺����
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

// @brief ����ͼƬ�������ݵ�������
// @param nvinfer_ptr NvinferStruct�ṹ��ָ��
// @para node_name_wchar ����ڵ�����
// @param image_data ͼƬ��������
// @param image_size ͼƬ���ݳ���
// @return NvinferStruct�ṹ��ָ��
extern "C"  __declspec(dllexport) void* __stdcall load_image_data(void* nvinfer_ptr,
	const wchar_t* node_name_wchar, uchar * image_data, int img_w, int img_h, int BN_means) {
	// �ع�NvinferStruct
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;

	// ��ȡ����ڵ���Ϣ
	const char* node_name = wchar_to_char(node_name_wchar);
	int node_index = p->engine->getBindingIndex(node_name);
	// ��ȡ����ڵ�δ����Ϣ
	int INPUT_W = p->input_w;
	int INPUT_H = p->input_h;
	int INPUT_C = p->input_c;
	// ����ڵ��ά��״
	cv::Size node_shape(INPUT_W, INPUT_H);
	// ����ڵ��ά��С
	size_t node_data_length = INPUT_W * INPUT_H * INPUT_C;

	// Ԥ������������
	cv::Mat input_image = data_to_mat(image_data, img_w, img_h);
	uchar* img_data = input_image.data;
	int img_c = input_image.channels();

	if (BN_means == 0)
	{
		std::vector<float> input_data(node_data_length);
		// ��ͼ���һ������������ָ����С
		input_image = cv::dnn::blobFromImage(input_image, 1 / 255.0, node_shape, cv::Scalar(0, 0, 0), true, false);
		// ��ͼƬ����copy����������
		memcpy(input_data.data(), input_image.ptr<float>(), node_data_length * sizeof(float));

		// ����cuda��
		cudaStreamCreate(&p->stream);

		// �������������ڴ浽GPU�Դ�
		cudaMemcpyAsync(p->data_buffer[node_index], input_data.data(), node_data_length * sizeof(float), cudaMemcpyHostToDevice, p->stream);

		std::vector<float>().swap(input_data);
		input_image.release();
	}
	else if (BN_means == 1)
	{
		std::vector<float> input_data(node_data_length);
		cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB); // ��ͼƬͨ���� BGR תΪ RGB
		// ������ͼƬ����tensor����Ҫ���������
		cv::resize(input_image, input_image, node_shape, 0, 0, cv::INTER_CUBIC);
		// ͼ�����ݹ�һ��
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
		// ����cuda��
		cudaStreamCreate(&p->stream);

		// �������������ڴ浽GPU�Դ�
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
		// �������������ڴ浽GPU�Դ�
		cudaMemcpyAsync(p->data_buffer[node_index], gpu_data_planes, INPUT_C * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyDeviceToDevice, p->stream);
		CHECK(cudaFree(gpu_img_buf));
		CHECK(cudaFree(gpu_img_resize_buf));
		CHECK(cudaFree(gpu_data_buf));
		CHECK(cudaFree(gpu_data_planes));
	}
	return (void*)p;

}


// @brief ģ������
// @param nvinfer_ptr NvinferStruct�ṹ��ָ��
// @return NvinferStruct�ṹ��ָ��
extern "C"  __declspec(dllexport) void* __stdcall infer(void* nvinfer_ptr) {
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	// ģ������
	p->context->enqueueV2(p->data_buffer, p->stream, nullptr);
	return (void*)p;
}


// @brief ��ȡ��������
// @param nvinfer_ptr NvinferStruct�ṹ��ָ��
// @para node_name_wchar ����ڵ�����
// @param output_result �������ָ��
extern "C"  __declspec(dllexport) void __stdcall read_infer_result(void* nvinfer_ptr,
	const wchar_t* node_name_wchar, int* output_result, int NMS_means) {
	
	// �ع�NvinferStruct
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	int feature_dim = p->feature_dim;
	int class_num = p->class_num;
	size_t node_data_length = feature_dim * class_num;

	// ��ȡ����ڵ���Ϣ
	const char* node_name = wchar_to_char(node_name_wchar);
	int node_index = p->engine->getBindingIndex(node_name);

	if (NMS_means==0)
	{
		// ��ȡ�������
		// �����������
		float* result = new float[node_data_length];
		// �����������GPU�Դ浽�ڴ�
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
			// ��ȡһ�����������ֵ����λ��
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
			// ���Ŷ� 0��1֮��
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

// @brief ɾ���ڴ��ַ
// @param nvinfer_ptr NvinferStruct�ṹ��ָ��
extern "C"  __declspec(dllexport) void __stdcall nvinfer_delete(void* nvinfer_ptr) {
	NvinferStruct* p = (NvinferStruct*)nvinfer_ptr;
	delete p->data_buffer;
	p->context->destroy();
	p->engine->destroy();
	p->runtime->destroy();
	delete p;
}