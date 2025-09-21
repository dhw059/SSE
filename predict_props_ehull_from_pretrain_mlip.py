import torch
import numpy as np
import json
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
# import matgl
import json
from pymatgen.core.structure import Structure


# from matcalc.relaxation import RelaxCalc
# from matcalc.utils import get_universal_calculator

from matcalc import ElasticityCalc, EOSCalc, PESCalculator, PhononCalc, RelaxCalc

# from pymatgen.ext.matproj import MPRester
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

device = "cuda" if torch.cuda.is_available() else "cpu"


# 参数设置
fmax = 1000000000000000000.0
# opt = "BFGSLineSearch"

# model_path = "/home/deep/mattersim/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
# calculator = MatterSimCalculator(load_path=model_path, device=device)

# calculator = get_universal_calculator("sevennet")
calculator = PESCalculator.load_universal('grace')

# 加载数据
# with open("/home/datasets/sample_all_materials_data_ehull_processed/sample_all_materials_data_ehull_processed.json", "r") as f:
with open("/home/datasets/li_sse_ehull_mp_datasest/li_sse_ehull_mp_datasest.json", "r") as f:
# with open("/home/datasets/include_li_mp_ehull/include_li_mp_ehull.json", "r") as f:

# with open("/home/datasets/li_sse_eform_mp_datasest/li_sse_eform_mp_datasest.json", "r") as f:
# with open("/home/datasets/include_li_mp_eform/include_li_mp_eform.json", "r") as f:
# with open("/home/datasets/sample_all_materials_data_form_processed/sample_all_materials_data_form_processed.json", "r") as f:
    exp_band_gap_data = json.load(f)


# 提取材料ID和数据
material_ids = exp_band_gap_data["index"]
data = exp_band_gap_data["data"]

# 确保 temps_pickle 目录存在
temps_pickle_dir = 'temps_pickle_mattersim'
# temps_pickle_dir = 'temps_pickle'
if not os.path.exists(temps_pickle_dir):
    os.makedirs(temps_pickle_dir)

# 初始化预测值和真实值列表
predictions = []
true_labels = []
# rester = MPRester('iTPrDnB1NuSywGXI')
rester = MPRester('iAihdZzrZYLQKZms1S43De90NiNK6ABB')
relax_calc = RelaxCalc(calculator, fmax=fmax, relax_cell=False, relax_atoms=False)
corrections = {}

# 检查 corrections.json 文件是否存在
if os.path.exists('corrections.json'):
    print("Corrections file found. Loading existing corrections...")
    with open('corrections.json', 'r') as f:
        corrections = json.load(f)
        

# 遍历数据，提取结构信息并进行预测
for idx, entry in tqdm(enumerate(data), total=len(data)):
    struct_info = entry[0]
    exp_band_gap = entry[1]
    # 将结构信息转换为 pymatgen.Structure 对象
    struct = Structure.from_dict(struct_info)
    # 检查结构是否为有序结构
    if struct.is_ordered:
        try:
            comp = struct.composition
            elements = sorted(comp.elements, key=lambda el: el.symbol)
            chemsys = "-".join([el.symbol for el in elements])

            # 构建 pickle 文件路径
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
                print(f"ValueError during relaxation for material ID {material_ids[idx]}: {e}")
                continue
            except RuntimeError as e:
                print(f"RuntimeError during relaxation for material ID {material_ids[idx]}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error during relaxation for material ID {material_ids[idx]}: {e}")
                continue

            material_id = material_ids[idx]
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
                    print(f"IndexError when retrieving correction_per_atom for material ID {material_ids[idx]}: {e}")
                    correction_per_atom = 0.0
                except Exception as e:
                    print(f"Unexpected error when retrieving correction_per_atom for material ID {material_ids[idx]}: {e}")
                    correction_per_atom = 0.0

            entry = ComputedStructureEntry(final_structure, energy, correction=correction_per_atom * final_structure.num_sites)
            entries = [entry] + mp_entries

            pd = PhaseDiagram(entries)
            # 计算形成能
            # form_energy_per_atom = pd.get_form_energy_per_atom(entry)
            ehull = pd.get_e_above_hull(entry) 

            predictions.append(float(ehull))
            true_labels.append(exp_band_gap)
        except ValueError as e:
            # 如果遇到不支持的元素，记录错误并跳过该结构
            print(f"Skipping structure {struct.composition.reduced_formula}: {e}")
            continue

# 保存 corrections 字典为 JSON 文件
with open('corrections.json', 'w') as f:
    json.dump(corrections, f)

# 转换为 numpy 数组
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# 过滤掉包含 NaN 的数据
valid_indices = ~np.isnan(predictions) & ~np.isnan(true_labels)
predictions = predictions[valid_indices]
true_labels = true_labels[valid_indices]

# 计算 MAE 和 RMSE
mae = np.mean(np.abs(predictions - true_labels))
rmse = np.sqrt(mean_squared_error(true_labels, predictions))
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")



def plot_predict_true(y_predict, y_true, data_unit: str = "", model_name: str = "",
                      filepath: str = "", file_name: str = "", dataset_name: str = "", target_names: str = "",
                      figsize: list = None, dpi: float = None, show_fig: bool = False):
    r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.

    Args:
        y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        data_unit (str): Name of the data's unit.
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        target_names (str): Name of the targets.
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.

    Returns:
        matplotlib.pyplot.figure: Figure of the scatter plot.
    """
    if len(y_predict.shape) == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    if data_unit is None:
        data_unit = ""
    if isinstance(data_unit, str):
        data_unit = [data_unit] * num_targets
    if len(data_unit) != num_targets:
        print("WARNING:kgcnn: Targets do not match units for plot.")
    if target_names is None:
        target_names = ""
    if isinstance(target_names, str):
        target_names = [target_names] * num_targets
    if len(target_names) != num_targets:
        print("WARNING:kgcnn: Targets do not match names for plot.")

    if figsize is None:
        figsize = [6, 5]
    if dpi is None:
        dpi = 300.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(num_targets):
        delta_valid = y_true[:, i] - y_predict[:, i]
        delta_valid_value = delta_valid[~np.isnan(delta_valid)]

        mae_valid = np.mean(np.abs(delta_valid_value))
        rmse_valid = np.sqrt(np.mean(delta_valid_value ** 2))
        label = f"{target_names[i]} MAE: {mae_valid:.4f}, RMSE: {rmse_valid:.4f} [{data_unit[i]}]"

        plt.scatter(y_predict[:, i], y_true[:, i], label=label)

    min_max = np.amin(y_true[~np.isnan(y_true)]).astype("float"), np.amax(y_true[~np.isnan(y_true)]).astype("float")
    plt.plot(np.arange(*min_max, 0.05), np.arange(*min_max, 0.05), color='red')
    plt.xlabel('Predicted value')
    plt.ylabel('DFT value')
    plt.title("Prediction of " + model_name + " for " + dataset_name)
    plt.legend(loc='upper left', fontsize='medium')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig


# 绘制预测结果图并保存
data_unit = 'eV/atom'
# data_unit = 'Gpa'
model_name = 'sevennet'

# dataset_name = 'SampleAllMPFormDataset'
# dataset_name =  'MPBulkModulusVrhDataset'
# dataset_name = 'ContainLiCompoundsDataset'
dataset_name = 'SampleAllMPEhullDataset'
# dataset_name = 'Li-SSE-ShearModulusMPDataset'
# dataset_name = 'ICSDExperimentDataset'
# dataset_name = 'Li-SSE-FormationenergyMPDataset'
# dataset_name = 'Li-SSE-EabovehullMPDataset'

target_names = 'Energy above hull'
# target_names = 'Bulk modulus vrh'
# target_names = 'Formation energy'
file_name = "predict_fold.png"
filepath = "./"

plot_predict_true(predictions, true_labels, data_unit=data_unit, model_name=model_name, 
                  filepath=filepath, file_name=file_name, dataset_name=dataset_name, target_names=target_names, show_fig=True)

# 保存 predictions 和 true_labels 为 JSON 文件
output_filename = f"{model_name}_{dataset_name}_{target_names}.json"
with open(output_filename, 'w') as f:
    json.dump({'predictions': predictions.tolist(), 'true_labels': true_labels.tolist()}, f)
