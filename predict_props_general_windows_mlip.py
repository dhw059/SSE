import torch
import numpy as np
import json
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

import json
from pymatgen.core.structure import Structure,Element


from matcalc import ElasticityCalc, EOSCalc, PESCalculator, PhononCalc, RelaxCalc


from mp_api.client import MPRester
from pymatgen.entries.computed_entries import (
    ComputedEntry,
    ComputedStructureEntry,)
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.phase_diagram import (
    CompoundPhaseDiagram,
    PDPlotter,
    PhaseDiagram,
)
import pickle
from tqdm import tqdm
# from mattersim.forcefield import MatterSimCalculator

import torch._dynamo
torch._dynamo.config.suppress_errors = True

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_structure_from_cif(cif_content):
    """从 CIF 内容加载结构文件并返回结构和组成信息"""
    structure = Structure.from_str(cif_content, fmt='cif')
    comp = structure.composition
    return structure, comp

def calculate_electrochemical_stability(pd, entry, entries):
    """计算电化学稳定性"""
    li_entries = [e for e in entries if e.composition.reduced_formula == "Li"]
    uli0 = min(li_entries, key=lambda e: e.energy_per_atom).energy_per_atom

    el_profile = pd.get_element_profile(Element("Li"), entry.composition)
    voltages, reactions, evolutions = [], [], []

    for d in el_profile:
        voltage = -(d["chempot"] - uli0)
        voltages.append(voltage)
        reactions.append(d["reaction"])
        evolutions.append(d["evolution"])

    reduction_potential = min(voltages, key=lambda v: abs(evolutions[voltages.index(v)]))
    # oxidation_potential = next((v for v, e in zip(voltages, evolutions) if e < -0.2), None)
    oxidation_potential = next((v for v, e in zip(voltages, evolutions) if e < -0.2), float('nan'))

    electrochemical_stability_window = (oxidation_potential - reduction_potential) if reduction_potential is not None and oxidation_potential is not None else 0
    return reduction_potential, oxidation_potential, electrochemical_stability_window


# 参数设置
fmax = 1e20
# model_path = "/home/deep/mattersim/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
# calculator = MatterSimCalculator(load_path=model_path, device=device)

calculator = PESCalculator.load_universal('orb')

# 加载数据
json_path = "/home/deep/gcnn_keras-master/data/search_from_mp/eform_ehull/contain_li_not_in_sample_mp_form_ehull_data_no_none_with_properties.json"
with open(json_path, "r") as f:
    json_data = json.load(f)

# 确保 temps_pickle 目录存在
temps_pickle_dir = 'temps_pickle_mattersim'
if not os.path.exists(temps_pickle_dir):
    os.makedirs(temps_pickle_dir)

# 初始化预测值和真实值列表
predictions_r = []
true_labels_r = []
predictions_o = []
true_labels_o = []
predictions_w = []
true_labels_w = []

# rester = MPRester('iTPrDnB1NuSywGXI')
rester = MPRester('iAihdZzrZYLQKZms1S43De90NiNK6ABB')
relax_calc = RelaxCalc(calculator, fmax=fmax,  relax_cell=False, relax_atoms=False)
corrections = {}

# 检查 corrections.json 文件是否存在
if os.path.exists('corrections.json'):
    print("Corrections file found. Loading existing corrections...")
    with open('corrections.json', 'r') as f:
        corrections = json.load(f)
        

# 遍历数据，提取结构信息并进行预测
# for idx, entry in tqdm(enumerate(data), total=len(data)):

# for i, (mp_id, data) in enumerate(json_data.items()):
for i, (mp_id, data) in enumerate(tqdm(json_data.items(), total=len(json_data), desc="Processing Materials")):
   
    cif_content = data.get("cif", "")
    r_true = data.get("reduction_potential")
    o_true = data.get("oxidation_potential")
    w_true = data.get("electrochemical_stability_window")

    # 跳过任意字段为 None 的情况
    if None in (r_true, o_true, w_true):
        continue

    # if w_true <= 5.0:
    #     continue

    # 将结构信息转换为 pymatgen.Structure 对象
    if cif_content:
        struct, comp = load_structure_from_cif(cif_content)
    # 检查结构是否为有序结构
    if struct.is_ordered:
        try:
            
            elements = sorted(comp.elements, key=lambda el: el.symbol)
            chemsys = "-".join([el.symbol for el in elements])
            pickle_file_path = os.path.join(temps_pickle_dir, f'{chemsys}.pkl')

            # 检查是否存在对应的 pickle 文件
            if os.path.exists(pickle_file_path):
                try:
                    with open(pickle_file_path, 'rb') as f:
                        mp_entries = pickle.load(f)
                except FileNotFoundError as e:
                    print(f"FileNotFoundError when loading pickle file for chemsys {chemsys}: {e}")
                    mp_entries = []
                except Exception as e:
                    print(f"Unexpected error when loading pickle file for chemsys {chemsys}: {e}")
                    mp_entries = []
            else:
                try:
                    mp_entries = rester.get_entries_in_chemsys(chemsys)
                    # 保存为 pickle 文件
                    with open(pickle_file_path, 'wb') as f:
                        pickle.dump(mp_entries, f)
                except Exception as e:
                    print(f"Failed to retrieve entries for chemsys {chemsys}: {e}")
                    mp_entries = []

            # 结构弛豫计算
            try:
                relax_results = relax_calc.calc(struct)
                final_structure = relax_results["final_structure"]
                energy = relax_results["energy"]
            except ValueError as e:
                print(f"ValueError during relaxation for material ID {mp_id}: {e}")
                continue
            except RuntimeError as e:
                print(f"RuntimeError during relaxation for material ID {mp_id}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error during relaxation for material ID {mp_id}: {e}")
                continue

            material_id = mp_id
            # 检查 corrections 字典中是否已有该 material_id 的 correction_per_atom
            if material_id in corrections:
                correction_per_atom = corrections[material_id]
            else:
                try:
                    mp_entry = rester.get_entries(material_id)
                    correction_per_atom = mp_entry[0].correction_per_atom
                    # 更新 corrections 字典
                    corrections[material_id] = correction_per_atom
                except IndexError as e:
                    print(f"IndexError when retrieving correction_per_atom for material ID {mp_id}: {e}")
                    correction_per_atom = 0.0
                except Exception as e:
                    print(f"Unexpected error when retrieving correction_per_atom for material ID {mp_id}: {e}")
                    correction_per_atom = 0.0

            entry = ComputedStructureEntry(final_structure, energy, correction=correction_per_atom * final_structure.num_sites)
            # entry = ComputedStructureEntry(final_structure, energy )
            
            entries = [entry] + mp_entries

            pd = PhaseDiagram(entries)
            # reduction_potential, oxidation_potential and electrochemical_stability_window
            r_pred, o_pred, w_pred= calculate_electrochemical_stability(pd, entry, entries)

            # 增加判断，避免 None 或 NaN 被加入列表
            if None in (r_pred, o_pred, w_pred):
                continue
            if np.isnan([r_pred, o_pred, w_pred]).any():
                continue
        
            predictions_r.append(float(r_pred))
            true_labels_r.append(r_true)
            predictions_o.append(float(o_pred))
            true_labels_o.append(o_true)
            predictions_w.append(float(w_pred))
            true_labels_w.append(w_true)

        except ValueError as e:
            # 如果遇到不支持的元素，记录错误并跳过该结构
            print(f"Skipping structure {struct.composition.reduced_formula}: {e}")
            continue

# 保存 corrections 字典为 JSON 文件
with open('corrections.json', 'w') as f:
    json.dump(corrections, f)


def plot_predict_true(y_predict, y_true, data_unit="V", model_name="MACE",
                      dataset_name="ContainLiCompounds", file_path="./", target_name="Reduction Potential"):
    """
    绘制预测值 vs 真实值图，显示 MAE 和 RMSE，使用文本标签而非图例。
    """
    y_predict = np.array(y_predict)
    y_true = np.array(y_true)

    # 过滤 NaN
    valid = ~np.isnan(y_predict) & ~np.isnan(y_true)
    y_predict = y_predict[valid]
    y_true = y_true[valid]

    # 计算指标
    mae = np.mean(np.abs(y_predict - y_true)) if len(y_predict) > 0 else float('nan')
    rmse = np.sqrt(mean_squared_error(y_true, y_predict)) if len(y_predict) > 0 else float('nan')

    # 创建绘图
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # 绘制散点图
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
    label_text = f"{target_name} ({data_unit})\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
    ax.text(0.05, 0.95, label_text,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 保存图像
    save_path = os.path.join(file_path, f"{model_name}_{dataset_name}_{target_name.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


# 绘制预测结果图并保存
data_unit = 'V'

# model_name="GRACE-2L-OAM"
# model_name = "MACE-MPA"
# model_name = "SevenNet-MF-ompa"
# model_name = "ORB v3"
# model_name = "DPA-3.1-3M"

# model_name = "MACE"
# model_name = "SevenNet"
model_name = "ORB v2"
# model_name = "MatterSim"
# model_name = "M3GNet"
# model_name = "CHGNet"

# model_name = "EquiformerV2"
# model_name = "TensorNet"

dataset_name = 'ContainLiCompounds'

# Plot each property
plot_predict_true(predictions_r, true_labels_r, data_unit=data_unit, model_name=model_name,
                  dataset_name=dataset_name, target_name="Reduction Potential")

plot_predict_true(predictions_o, true_labels_o, data_unit=data_unit, model_name=model_name,
                  dataset_name=dataset_name, target_name="Oxidation Potential")

plot_predict_true(predictions_w, true_labels_w, data_unit=data_unit, model_name=model_name,
                  dataset_name=dataset_name, target_name="Stability Window")

# Save results
output_filename = f"{model_name}_{dataset_name}_results.json"
with open(output_filename, 'w') as f:
    json.dump({
        'reduction': {'predictions': predictions_r, 'true': true_labels_r},
        'oxidation': {'predictions': predictions_o, 'true': true_labels_o},
        'window': {'predictions': predictions_w, 'true': true_labels_w}
    }, f)




# # Plotting function with updated label formatting
# def plot_predict_true(y_predict, y_true, data_unit="", model_name="",
#                       filepath="./", file_name="result", dataset_name="",
#                       target_names="", show_fig=True):
#     y_predict = np.array(y_predict)
#     y_true = np.array(y_true)
#     if len(y_predict.shape) == 1:
#         y_predict = y_predict.reshape(-1, 1)
#     if len(y_true.shape) == 1:
#         y_true = y_true.reshape(-1, 1)

#     num_targets = y_true.shape[1]
#     data_unit = [data_unit] * num_targets if isinstance(data_unit, str) else data_unit
#     target_names = [target_names] * num_targets if isinstance(target_names, str) else target_names

#     fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
#     for i in range(num_targets):
#         valid = ~np.isnan(y_true[:, i]) & ~np.isnan(y_predict[:, i])
#         y_t = y_true[valid, i]
#         y_p = y_predict[valid, i]

#         mae = np.mean(np.abs(y_t - y_p)) if len(y_t) > 0 else float('nan')
#         rmse = np.sqrt(mean_squared_error(y_t, y_p)) if len(y_t) > 0 else float('nan')

#         label = f"{target_names[i]} (V)\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
#         ax.scatter(y_p, y_t, label=label)

#     lims = [np.min(y_true[~np.isnan(y_true)]), np.max(y_true[~np.isnan(y_true)])]
#     ax.plot(lims, lims, color='red')
#     ax.set_xlabel('Predicted value')
#     ax.set_ylabel('DFT value')
#     ax.set_title(f"Prediction of {model_name} for {dataset_name}")
#     # ax.legend(loc='upper left', fontsize='large', bbox_to_anchor=(0.5, 1.0), fancybox=True, shadow=True)
#     ax.legend(loc='upper left', fontsize='large')

#     save_path = os.path.join(filepath, f"{model_name}_{dataset_name}_{file_name}.png")
#     plt.savefig(save_path, bbox_inches='tight')
#     if show_fig:
#         plt.show()
#     plt.close()

