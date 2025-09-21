import torch
import numpy as np
import json
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import time
from pymatgen.core.structure import Structure
from matcalc import PESCalculator, RelaxCalc

from mp_api.client import MPRester
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
import pickle
from tqdm import tqdm
# from mattersim.forcefield import MatterSimCalculator

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# 参数设置
fmax = 1e20
name = 'mattersim'
calculator = PESCalculator.load_universal(name)

# model_path = "/home/deep/mattersim/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
# calculator = MatterSimCalculator(load_path=model_path, device=device)

# 加载数据
with open("/home/datasets/include_li_mp_ehull/include_li_mp_ehull.json", "r") as f:
# with open("/home/datasets/li_sse_ehull_mp_datasest/li_sse_ehull_mp_datasest.json", "r") as f:
# with open("/home/datasets/sample_all_materials_data_ehull_processed/sample_all_materials_data_ehull_processed.json", "r") as f:

# with open("/home/datasets/li_sse_eform_mp_datasest/li_sse_eform_mp_datasest.json", "r") as f:   
# with open("/home/datasets/sample_all_materials_data_form_processed/sample_all_materials_data_form_processed.json", "r") as f:
# with open("/home/datasets/include_li_mp_eform/include_li_mp_eform.json", "r") as f:

# with open("/home/deep/gcnn_keras-master/filter/all_mp_data.json", "r") as f:
    exp_band_gap_data = json.load(f)

material_ids = exp_band_gap_data["index"]
data = exp_band_gap_data["data"]


# 从新的数据格式中提取信息
# material_ids = list(exp_band_gap_data.keys())
# data = list(exp_band_gap_data.values())


temps_pickle_dir = 'temps_pickle_mattersim'
os.makedirs(temps_pickle_dir, exist_ok=True)

# 初始化预测值和真实值列表
predictions = []
true_labels = []

rester = MPRester('iAihdZzrZYLQKZms1S43De90NiNK6ABB')
relax_calc = RelaxCalc(calculator, fmax=fmax, relax_cell=False, relax_atoms=False)
# relax_calc = RelaxCalc(calculator, relax_cell=False, relax_atoms=False)
corrections = {}

# 检查 corrections.json 文件是否存在
if os.path.exists('corrections.json'):
    with open('corrections.json', 'r') as f:
        corrections = json.load(f)

start_time = time.time()  # Record the start time
# 遍历数据
for idx, entry in tqdm(enumerate(data), total=len(data)):
# for idx, (material_id, entry_data) in tqdm(enumerate(zip(material_ids, data)), total=len(data)):
    struct_info = entry[0]
    true_ehull = entry[1]

    # 直接从数据中获取结构和能量信息
    # true_ehull = entry_data["energy_above_hull"]

    if np.isnan(true_ehull):
        continue

    # 过滤 ehull > 3.0 的结构
    # if true_ehull <= 4.0:
    #     continue

    # # 解析CIF字符串为结构对象
    # try:
    #     struct = Structure.from_str(entry_data["cif"], fmt="cif")
    # except Exception:
    #     # 如果CIF解析失败，尝试从其他字段重建结构
    #     continue

    struct = Structure.from_dict(struct_info)
    if not struct.is_ordered:
        continue

    try:
        comp = struct.composition
        elements = sorted(comp.elements, key=lambda el: el.symbol)
        chemsys = "-".join([el.symbol for el in elements])

        # 构建 pickle 文件路径
        pickle_file_path = os.path.join(temps_pickle_dir, f'{chemsys}.pkl')

        # 加载或获取 entries
        if os.path.exists(pickle_file_path):
            try:
                with open(pickle_file_path, 'rb') as f:
                    mp_entries = pickle.load(f)
            except Exception:
                mp_entries = rester.get_entries_in_chemsys(chemsys)
        else:
            mp_entries = rester.get_entries_in_chemsys(chemsys)
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(mp_entries, f)

        # 结构弛豫
        relax_results = relax_calc.calc(struct)
        final_structure = relax_results["final_structure"]
        energy = relax_results["energy"]

        material_id = material_ids[idx]
        if material_id in corrections:
            correction_per_atom = corrections[material_id]
        else:
            try:
                mp_entry = rester.get_entries(material_id)
                correction_per_atom = mp_entry[0].correction_per_atom
                corrections[material_id] = correction_per_atom
            except Exception:
                correction_per_atom = 0.0

        entry_obj = ComputedStructureEntry(final_structure, energy,
                                          correction=correction_per_atom * final_structure.num_sites)
        
        # entry_obj = ComputedStructureEntry(final_structure, energy,)
        entries = [entry_obj] + mp_entries

        pd = PhaseDiagram(entries)
        # ehull = pd.get_form_energy_per_atom(entry_obj)
        ehull = pd.get_e_above_hull(entry_obj)

        predictions.append(float(ehull))
        true_labels.append(true_ehull)

    except Exception as e:
        print(f"Error processing {material_ids[idx]}: {e}")


end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Loop completed in {elapsed_time:.2f} seconds")

# 保存 corrections
with open('corrections.json', 'w') as f:
    json.dump(corrections, f)

# 转换为 numpy 数组
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# 过滤 NaN
valid = ~np.isnan(predictions) & ~np.isnan(true_labels)
predictions = predictions[valid]
true_labels = true_labels[valid]

# 计算 MAE 和 RMSE
mae = np.mean(np.abs(predictions - true_labels))
rmse = np.sqrt(mean_squared_error(true_labels, predictions))
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")


# model_name="GRACE-2L-OAM"
# model_name = "MACE-MPA"
# model_name = "SevenNet-MF-ompa"
# model_name = "ORB v3"
# model_name = "DPA-3.1-3M"
# model_name = "TensorNet"

# model_name = "MACE"
# model_name = "SevenNet"
# model_name = "ORB v2"
model_name = "MatterSim"
# model_name = "M3GNet"
# model_name = "CHGNet"


dataset_name="ContainLiCompounds"
# dataset_name="UniversalCompounds"
# dataset_name="Li-SSE Compounds"

# dataset_name="MatterSim VS DFT" 
data_unit="eV/atom"


with open(f"{name}_{dataset_name}_loop_execution_time.txt", "w") as f:
    f.write(f"Loop completed in {elapsed_time:.2f} seconds\n")

def plot_predict_true_high_ehull(y_predict, y_true, data_unit=data_unit, model_name=model_name,
                                dataset_name= dataset_name, file_path="./"):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    ax.scatter(y_predict, y_true, alpha=0.7, label="Data Points")

    # 设置坐标轴和标题
    ax.set_xlabel("Predicted value", fontsize=20)
    ax.set_ylabel("DFT value", fontsize=20)
    ax.set_title(f"{model_name} for {dataset_name}", fontsize=18)
     # 增大刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 添加 y=x 线
    lims = [np.min([y_predict, y_true]), np.max([y_predict, y_true])]
    ax.plot(lims, lims, 'r--', linewidth=1.5)

    # 添加 MAE 和 RMSE 标签（两行显示）
    label_text = f"Energy above hull (eV/atom)\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
    # label_text = f"Formation per atom (eV/atom)\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
    ax.text(0.05, 0.95, label_text,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 保存图像
    save_path = os.path.join(file_path, f"{model_name}_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


# 绘图
plot_predict_true_high_ehull(predictions, true_labels)

# 保存结果
output_filename = f"{model_name}_{dataset_name}_results.json"
with open(output_filename, 'w') as f:
    json.dump({'predictions': predictions.tolist(), 'true_labels': true_labels.tolist()}, f)
