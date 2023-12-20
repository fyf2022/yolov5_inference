using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TensorRtSharp
{
    internal class NativeMethods
    {
        private const string tensorrt_dll_path = @"D:\vs_code\Inference-master\dll_import\tensorrt\tensorrtsharp.dll";

        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void onnx_to_engine(string onnx_file_path, string engine_file_path, int type);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr nvinfer_init(string engine_filename, int num_ionode, string input_node_name, string output_node_name);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void read_model_info(IntPtr nvinfer_ptr, string input_node_name, string output_node_name, ref int model_info);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr creat_gpu_buffer(IntPtr nvinfer_ptr, string node_name, ulong data_length);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr load_image_data(IntPtr nvinfer_ptr, string node_name, ref byte image_data, int image_Cols, int image_Rows, int BN_means);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static IntPtr infer(IntPtr nvinfer_ptr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void read_infer_result(IntPtr nvinfer_ptr, string node_name_wchar, ref int result, int NMS_means);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static void nvinfer_delete(IntPtr nvinfer_ptr);
    }
}
