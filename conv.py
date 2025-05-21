import os
import shutil
import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, shape_inference, numpy_helper
from huggingface_hub import hf_hub_download
from rknn.api import RKNN
import onnxruntime as rt 

MODEL_DIR = "vosk_models_onnx"
CALIB_DIR = "vosk_calibration_data"
RKNN_MODEL_DIR = "vosk_models_rknn"
REPO_ID = "alphacep/vosk-model-ru"
SUBFOLDER = "am-onnx"

ONNX_FILES = { "encoder": "encoder.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx",}
ONNX_INPUT_NAMES = { "encoder": ["x", "x_lens"], "decoder": ["y"], "joiner": ["encoder_out", "decoder_out"],}
RKNN_INPUT_SHAPES = { "encoder": [[1, 1, 80, 103], [1]], "decoder": [[1, 2]], "joiner": [[1, 512], [1, 512]]}
ENCODER_BATCH_FOR_ONNX_INPUT = RKNN_INPUT_SHAPES["encoder"][0][0] 
ENCODER_CHANNEL_FOR_ONNX_INPUT = RKNN_INPUT_SHAPES["encoder"][0][1] 
ENCODER_FEATURE_DIM_FOR_ONNX_INPUT = RKNN_INPUT_SHAPES["encoder"][0][2]   
ENCODER_TIME_LEN_FOR_ONNX_INPUT = RKNN_INPUT_SHAPES["encoder"][0][3] 

RKNN_CONFIG_PARAMS = {
    "encoder": {"mean_values": [], "std_values" : [], "quantized_dtype": "asymmetric_quantized-8", "quantized_algorithm": "normal", "target_platform": "rk3588",},
    "decoder": {"mean_values": [], "std_values" : [], "quantized_dtype": "asymmetric_quantized-8", "quantized_algorithm": "normal", "target_platform": "rk3588",},
    "joiner":  {"mean_values": [], "std_values" : [], "quantized_dtype": "asymmetric_quantized-8", "quantized_algorithm": "normal", "target_platform": "rk3588",}
}
NUM_CALIB_SAMPLES = 10
VOCAB_SIZE = 6254

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_downloaded = True
    for key, filename in ONNX_FILES.items():
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            # print(f"Downloading {filename} for {key}...")
            try:
                actual_file_path = hf_hub_download(repo_id=REPO_ID, filename=filename, subfolder=SUBFOLDER, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
                if actual_file_path != local_path:
                    final_dest_path = os.path.join(MODEL_DIR, filename)
                    if os.path.exists(final_dest_path) and final_dest_path != actual_file_path : os.remove(final_dest_path)
                    shutil.move(actual_file_path, final_dest_path)
            except Exception as e: print(f"Error downloading {filename}: {e}"); all_downloaded = False
    downloaded_subfolder_path = os.path.join(MODEL_DIR, SUBFOLDER)
    if os.path.exists(downloaded_subfolder_path) and os.path.isdir(downloaded_subfolder_path) and not os.listdir(downloaded_subfolder_path):
        try: os.rmdir(downloaded_subfolder_path)
        except OSError as e: print(f"Could not remove empty subfolder {downloaded_subfolder_path}: {e}")
    return all_downloaded

def check_and_get_onnx_io_names(model_path_or_proto, infer_shapes_locally=False, model_name_for_log="model"):
    try:
        if isinstance(model_path_or_proto, str):
            model_filename = os.path.basename(model_path_or_proto); model = onnx.load(model_path_or_proto)
        else: model_filename = f"loaded_{model_name_for_log}_object"; model = model_path_or_proto
        
        if infer_shapes_locally:
            try: model = shape_inference.infer_shapes(model)
            except Exception as e_si: print(f"  Local shape inference failed for {model_filename}: {e_si}")
        onnx.checker.check_model(model)
        input_names = [inp.name for inp in model.graph.input]
        input_types_info = []
        for inp in model.graph.input:
            type_name = TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
            dims = ["?"] 
            if inp.type.tensor_type.HasField("shape"):
                dims = [str(d.dim_value) if d.dim_value!=0 and d.dim_value is not None else (d.dim_param if d.dim_param else "Dyn") 
                         for d in inp.type.tensor_type.shape.dim]
            else: dims = ["Shape N/A"]
            input_types_info.append(f"{type_name}{dims}")
        # print(f"Checked Model: {model_filename} (Inputs: {input_types_info})")
        return input_names, None, model
    except Exception as e:
        print(f"Error loading or checking ONNX model '{model_path_or_proto if isinstance(model_path_or_proto, str) else model_name_for_log}': {e}")
        return None, None, None

def ensure_correct_conv_attributes(model):
    print("  Running ensure_correct_conv_attributes...")
    made_changes_overall = False
    for node_idx, node in enumerate(model.graph.node):
        if node.op_type != 'Conv':
            continue

        k_shape_attr = next((a for a in node.attribute if a.name == 'kernel_shape'), None)
        if not k_shape_attr: 
            # print(f"    WARNING: Conv node {node.name} has no kernel_shape! Skipping attr fix for it.")
            continue # Cannot determine kernel_rank without kernel_shape
        kernel_rank = len(k_shape_attr.ints)

        # 1. Dilations
        dil_attr = next((a for a in node.attribute if a.name == 'dilations'), None)
        current_dilations = list(dil_attr.ints) if dil_attr and hasattr(dil_attr, 'ints') else [] # Ensure .ints exists
        
        expected_dilations = [1] * kernel_rank 
        if kernel_rank == 2 and len(current_dilations) == 1: 
            expected_dilations = [1, current_dilations[0]]
        
        if list(current_dilations) != expected_dilations or not dil_attr:
            if dil_attr: node.attribute.remove(dil_attr)
            node.attribute.append(helper.make_attribute('dilations', expected_dilations))
            # if list(current_dilations) != expected_dilations : print(f"    Node {node.name}: CORRECTED dilations from {current_dilations} to {expected_dilations}.")
            # else: print(f"    Node {node.name}: ADDED default dilations {expected_dilations}.")
            made_changes_overall = True
        
        # 2. Strides
        str_attr = next((a for a in node.attribute if a.name == 'strides'), None)
        current_strides = list(str_attr.ints) if str_attr and hasattr(str_attr, 'ints') else []
        expected_strides = [1] * kernel_rank
        if kernel_rank == 2 and len(current_strides) == 1: 
            expected_strides = [1, current_strides[0]]

        if list(current_strides) != expected_strides or not str_attr:
            if str_attr: node.attribute.remove(str_attr)
            node.attribute.append(helper.make_attribute('strides', expected_strides))
            # if list(current_strides) != expected_strides: print(f"    Node {node.name}: CORRECTED strides from {current_strides} to {expected_strides}.")
            # else: print(f"    Node {node.name}: ADDED default strides {expected_strides}.")
            made_changes_overall = True

        # 3. Pads
        pads_attr = next((a for a in node.attribute if a.name == 'pads'), None)
        current_pads = list(pads_attr.ints) if pads_attr and hasattr(pads_attr, 'ints') else []
        auto_pad_val = next((helper.get_attribute_value(a) for a in node.attribute if a.name == 'auto_pad'), "NOTSET").upper()
        
        expected_pads_len = kernel_rank * 2
        if auto_pad_val == "NOTSET":
            if not pads_attr or len(current_pads) != expected_pads_len:
                new_pads = [0] * expected_pads_len 
                if kernel_rank == 2 and len(current_pads) == 2 and k_shape_attr.ints[0] == 1: # kH=1
                    new_pads = [0, current_pads[0], 0, current_pads[1]]
                
                if pads_attr: node.attribute.remove(pads_attr)
                node.attribute.append(helper.make_attribute("pads", new_pads)); made_changes_overall = True
                # if new_pads != current_pads : print(f"    Node {node.name}: SET pads to {new_pads}.")
        
        # 4. Group
        group_attr = next((a for a in node.attribute if a.name == 'group'), None)
        if not group_attr : node.attribute.append(helper.make_attribute("group", 1)); made_changes_overall = True
        elif group_attr.i < 1 : group_attr.i = 1; made_changes_overall = True
            
    if made_changes_overall: print("  ensure_correct_conv_attributes: Changes made to some Conv attributes.")
    return made_changes_overall

def convert_encoder_to_4d_nchw(model, encoder_x_input_name):
    print(f"Attempting deep conversion for encoder input '{encoder_x_input_name}' and Conv layers.")
    model_input_changed = False; conv_layers_adapted_count = 0

    x_input_value_info = next((inp for inp in model.graph.input if inp.name == encoder_x_input_name), None)
    if not x_input_value_info: print(f"  ERROR: Input '{encoder_x_input_name}' not found."); return False

    dims = x_input_value_info.type.tensor_type.shape.dim
    if len(dims) == 3:
        print(f"  Input '{encoder_x_input_name}' is 3D, converting to 4D NCHW definition.")
        N, C, H, W_time = (ENCODER_BATCH_FOR_ONNX_INPUT, ENCODER_CHANNEL_FOR_ONNX_INPUT, 
                           ENCODER_FEATURE_DIM_FOR_ONNX_INPUT, ENCODER_TIME_LEN_FOR_ONNX_INPUT)
        x_input_value_info.type.tensor_type.shape.ClearField("dim")
        for val in [N, C, H, W_time]: x_input_value_info.type.tensor_type.shape.dim.add().dim_value = val
        model_input_changed = True
    elif not (len(dims) == 4 and dims[1].dim_value == 1 and dims[0].dim_value == ENCODER_BATCH_FOR_ONNX_INPUT and \
              dims[2].dim_value == ENCODER_FEATURE_DIM_FOR_ONNX_INPUT and dims[3].dim_value == ENCODER_TIME_LEN_FOR_ONNX_INPUT):
        print(f"  Input '{encoder_x_input_name}' is {len(dims)}D but not matching target NCHW or C!=1. Aborting.")
        return False

    initializers_to_remove = []; initializers_to_add = []
    for node_idx, node in enumerate(model.graph.node):
        if node.op_type == "Conv":
            weight_name = node.input[1]
            weight_init = next((init for init in model.graph.initializer if init.name == weight_name), None)
            if not weight_init: continue
            W_array = numpy_helper.to_array(weight_init)

            if W_array.ndim == 3: 
                OC, IC_1D, kT = W_array.shape
                new_k_H, new_k_W = 1, kT 
                new_inC_conv2d = IC_1D 
                new_W_array = W_array.reshape(OC, new_inC_conv2d, new_k_H, new_k_W)
                initializers_to_remove.append(weight_init)
                initializers_to_add.append(numpy_helper.from_array(new_W_array.astype(W_array.dtype), name=weight_name))
                
                k_shape_attr = next((a for a in node.attribute if a.name == 'kernel_shape'), None)
                if k_shape_attr: node.attribute.remove(k_shape_attr)
                node.attribute.append(helper.make_attribute("kernel_shape", [new_k_H, new_k_W]))
                conv_layers_adapted_count +=1
    
    for old_init in initializers_to_remove: model.graph.initializer.remove(old_init)
    for new_init in initializers_to_add: model.graph.initializer.append(new_init)
    
    if model_input_changed or conv_layers_adapted_count > 0:
        print(f"  Deep conversion: Input def changed: {model_input_changed}, Conv1D layers adapted: {conv_layers_adapted_count}")
        return True
    return False

def simplify_onnx(model_or_path, simplified_model_path, overwrite_shapes_dict=None, skip_simplifier=False):
    if skip_simplifier:
        if isinstance(model_or_path, str):
            if model_or_path != simplified_model_path: shutil.copy(model_or_path, simplified_model_path)
        else: onnx.save(model_or_path, simplified_model_path)
        return True
    try:
        import onnxsim
        model_simplified, check = onnxsim.simplify(model_or_path, overwrite_input_shapes=overwrite_shapes_dict, perform_optimization=True)
        if check: onnx.save(model_simplified, simplified_model_path)
        else: onnx.save(model_simplified, simplified_model_path); print("Simplification check failed, but saved.")
        return True
    except ImportError:
        print("onnx-simplifier not installed. Skipping.");
        if isinstance(model_or_path, str): 
            if model_or_path != simplified_model_path: shutil.copy(model_or_path, simplified_model_path)
        else: onnx.save(model_or_path, simplified_model_path)
        return True
    except Exception as e:
        print(f"Error during ONNX simplification: {e}. Copying/saving original.");
        if isinstance(model_or_path, str):
            if model_or_path != simplified_model_path: shutil.copy(model_or_path, simplified_model_path)
        else: onnx.save(model_or_path, simplified_model_path)
        return False

def prepare_onnx_models():
    print("\n--- Preparing ONNX Models ---")
    prepared_paths = {}
    for key in ONNX_FILES.keys():
        original_path = os.path.join(MODEL_DIR, ONNX_FILES[key])
        prepared_path = os.path.join(MODEL_DIR, f"{key}_prepared.onnx")
        print(f"\n--- Processing model: {key} ---")
        if not os.path.exists(original_path): print(f"Original ONNX {original_path} not found. Skipping."); continue

        _, _, current_model_object = check_and_get_onnx_io_names(original_path, model_name_for_log=f"{key}_original")
        if not current_model_object: print(f"Could not load model {original_path}. Skipping."); continue
        
        ONNX_INPUT_NAMES[key] = [inp.name for inp in current_model_object.graph.input]
        
        if key == "encoder":
            if convert_encoder_to_4d_nchw(current_model_object, ONNX_INPUT_NAMES[key][0]):
                try: 
                    _, _, current_model_object = check_and_get_onnx_io_names(current_model_object, infer_shapes_locally=True, model_name_for_log=f"{key}_deep_conv_checked")
                    if not current_model_object : raise ValueError("Model invalid after deep_conv check")
                except Exception as e_conv: print(f"ERROR after deep conv for {key}: {e_conv}. Reverting."); _, _, current_model_object = check_and_get_onnx_io_names(original_path)

            ensure_correct_conv_attributes(current_model_object)
            
            try: 
                _, _, current_model_object = check_and_get_onnx_io_names(current_model_object, infer_shapes_locally=True, model_name_for_log=f"{key}_final_attr_checked")
                if not current_model_object: raise ValueError("Model invalid after final attr check")
            except Exception as e_final: print(f"ERROR on final check for {key}: {e_final}. Using potentially unverified model object.");
        
        elif key == "decoder":
            for inp_obj_idx, inp_onnx_obj in enumerate(current_model_object.graph.input):
                if inp_onnx_obj.name == ONNX_INPUT_NAMES[key][0]:
                    if inp_onnx_obj.type.tensor_type.elem_type == TensorProto.INT64:
                        current_model_object.graph.input[inp_obj_idx].type.tensor_type.elem_type = TensorProto.INT32
        
        target_shapes_for_sim = RKNN_INPUT_SHAPES[key]
        current_inputs_for_sim = ONNX_INPUT_NAMES[key]
        overwrite_shapes_for_sim = {current_inputs_for_sim[i]: target_shapes_for_sim[i] for i in range(len(current_inputs_for_sim))} \
                                   if len(current_inputs_for_sim) == len(target_shapes_for_sim) else None

        skip_simplifier = (key == "encoder")
        simplify_onnx(current_model_object, prepared_path, overwrite_shapes_for_sim, skip_simplifier=skip_simplifier)
        
        prepared_paths[key] = prepared_path
        print(f"--- Sanity check for saved file: {prepared_path} ---")
        try:
            sess_options = rt.SessionOptions(); sess_options.log_severity_level = 3
            # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL # Для чистой проверки
            rt.InferenceSession(prepared_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            print(f"  ONNX Runtime loaded {prepared_path} successfully.")
        except Exception as e_ort_load:
            print(f"  ERROR: ONNX Runtime failed to load {prepared_path}: {e_ort_load}")

        final_input_names_from_file, _, _ = check_and_get_onnx_io_names(prepared_path, infer_shapes_locally=True, model_name_for_log=f"{key}_prepared_final")
        if final_input_names_from_file: ONNX_INPUT_NAMES[key] = final_input_names_from_file
    return prepared_paths

def create_dummy_calibration_data():
    os.makedirs(CALIB_DIR, exist_ok=True); dataset_files = {}; print("\n--- Creating Dummy Calibration Data ---")
    for key, input_shape_list in RKNN_INPUT_SHAPES.items():
        dataset_file_path = os.path.join(CALIB_DIR, f"dataset_{key}.txt"); dataset_files[key] = dataset_file_path
        num_inputs = len(input_shape_list)
        # print(f"Creating data for {key} ({NUM_CALIB_SAMPLES} samples), {num_inputs} input(s) based on RKNN_INPUT_SHAPES: {input_shape_list}")
        with open(dataset_file_path, "w") as f:
            for i in range(NUM_CALIB_SAMPLES):
                npy_filenames = []
                for input_idx in range(num_inputs):
                    shape = tuple(input_shape_list[input_idx]); base_npy_fn = f"{key}_input{input_idx}_calib_{i}.npy"
                    full_npy_save_path = os.path.join(CALIB_DIR, base_npy_fn)
                    if key == "encoder": data = np.random.rand(*shape).astype(np.float32) if input_idx == 0 else np.array([ENCODER_TIME_LEN_FOR_ONNX_INPUT], dtype=np.int32).reshape(shape)
                    elif key == "decoder": data = np.random.randint(0, VOCAB_SIZE, size=shape, dtype=np.int32)
                    else: data = np.random.rand(*shape).astype(np.float32)
                    np.save(full_npy_save_path, data); npy_filenames.append(base_npy_fn)
                f.write(" ".join(npy_filenames) + "\n")
    return dataset_files

def convert_to_rknn(prepared_onnx_paths, calib_dataset_files):
    os.makedirs(RKNN_MODEL_DIR, exist_ok=True); print("\n--- Converting to RKNN ---")
    for key, onnx_path in prepared_onnx_paths.items():
        if not onnx_path or not os.path.exists(onnx_path): print(f"ONNX for {key} invalid/not found. Skipping."); continue
        print(f"\nProcessing {key} model: {onnx_path}"); rknn_path = os.path.join(RKNN_MODEL_DIR, f"{key}.rknn")
        
        rknn_env_set_for_key = False
        # if key == "encoder":
        #     print("Setting os.environ['RKNN_NO_ORT_FOLD_CONST'] = '1' for ENCODER")
        #     os.environ["RKNN_NO_ORT_FOLD_CONST"] = "1"
        #     rknn_env_set_for_key = True

        rknn = RKNN(verbose=True); # print("Configuring RKNN...");
        current_rknn_config = RKNN_CONFIG_PARAMS[key].copy()
        if rknn.config(**current_rknn_config) != 0: 
            print(f"Config RKNN for {key} failed."); 
            if rknn_env_set_for_key: del os.environ["RKNN_NO_ORT_FOLD_CONST"]
            rknn.release(); continue
        # print("RKNN config successful.")
        
        current_onnx_graph_input_names = ONNX_INPUT_NAMES[key] 
        current_rknn_target_shapes_for_inputs = RKNN_INPUT_SHAPES[key]
        
        # print(f"  Final check before rknn.load_onnx for '{key}':")
        # print(f"    RKNN load_onnx using ONNX file: {onnx_path}")
        # print(f"    RKNN load_onnx using input names from ONNX graph: {current_onnx_graph_input_names}")
        # print(f"    RKNN load_onnx using target shapes for these inputs: {current_rknn_target_shapes_for_inputs}")

        if len(current_onnx_graph_input_names) != len(current_rknn_target_shapes_for_inputs): 
            print(f"FATAL: Mismatch num names vs shapes for {key}."); 
            if rknn_env_set_for_key: del os.environ["RKNN_NO_ORT_FOLD_CONST"]
            rknn.release(); continue
        
        ret = rknn.load_onnx(model=onnx_path, inputs=current_onnx_graph_input_names, input_size_list=current_rknn_target_shapes_for_inputs)
        if ret != 0: 
            print(f"Error loading ONNX for {key}. Code: {ret}."); 
            if rknn_env_set_for_key: del os.environ["RKNN_NO_ORT_FOLD_CONST"]
            rknn.release(); continue
        # print("ONNX model loaded successfully.")

        calib_file_path = calib_dataset_files.get(key)
        if not calib_file_path or not os.path.exists(calib_file_path): 
            print(f"Calib file {calib_file_path} not found for {key}."); 
            if rknn_env_set_for_key: del os.environ["RKNN_NO_ORT_FOLD_CONST"]
            rknn.release(); continue
        # print(f"Building RKNN model for {key} with quantization (dataset: {calib_file_path})...")
        
        ret = rknn.build(do_quantization=True, dataset=calib_file_path)
        if ret != 0: 
            print(f"Error building RKNN for {key}. Code: {ret}."); 
            if rknn_env_set_for_key: del os.environ["RKNN_NO_ORT_FOLD_CONST"]
            rknn.release(); continue
        print(f"RKNN model for {key} built successfully.") # Успех только если дошли сюда

        # print(f"Exporting RKNN model to: {rknn_path}")
        if rknn.export_rknn(rknn_path) != 0: print(f"Error exporting RKNN for {key}.")
        # else: print(f"RKNN model for {key} exported to {rknn_path}.")
        
        if rknn_env_set_for_key: 
            # print("Unsetting os.environ['RKNN_NO_ORT_FOLD_CONST']")
            del os.environ["RKNN_NO_ORT_FOLD_CONST"]
        rknn.release()

if __name__ == "__main__":
    old_prepared_encoder = os.path.join(MODEL_DIR, "encoder_prepared.onnx")
    if os.path.exists(old_prepared_encoder):
        # print(f"Removing old {old_prepared_encoder}")
        os.remove(old_prepared_encoder)

    if not download_models(): print("Download failed. Exiting."); exit(1)
    prepared_onnx_files = prepare_onnx_models()
    if not prepared_onnx_files or not all(p and os.path.exists(p) for p in prepared_onnx_files.values()): print("ONNX preparation failed or some files are missing. Exiting."); exit(1)
    calibration_files = create_dummy_calibration_data()
    convert_to_rknn(prepared_onnx_files, calibration_files)
    print("\n--- Conversion process finished ---")
    # print(f"Source ONNX models in: {MODEL_DIR}"); print(f"Prepared ONNX models in: {MODEL_DIR} (*_prepared.onnx)")
    # print(f"Calibration data in: {CALIB_DIR}"); print(f"Converted RKNN models in: {RKNN_MODEL_DIR}")
