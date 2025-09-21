import torch
import json
import os
import warnings
import numpy as np
from pymatgen.core.structure import Structure
from sklearn.metrics import mean_squared_error
# from mattersim.forcefield import MatterSimCalculator
from time import perf_counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from matcalc import PESCalculator, PhononCalc, RelaxCalc


# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

warnings.filterwarnings("ignore", category=UserWarning, module="matgl")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_predict_true_high(y_predict, y_true, data_unit, model_name, dataset_name, text, file_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    ax.scatter(y_predict, y_true, alpha=0.7, label="Data Points")

    # 设置坐标轴和标题
    ax.set_xlabel(f"Predicted value", fontsize=20)
    ax.set_ylabel(f"DFT value", fontsize=20)
    ax.set_title(f"{model_name} for {dataset_name}", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # 添加 y=x 线
    lims = [np.min([y_predict, y_true]), np.max([y_predict, y_true])]
    ax.plot(lims, lims, 'r--', linewidth=1.5)

    # 添加 MAE 和 RMSE 标签
    mae = np.mean(np.abs(y_predict - y_true))
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    label_text = f"{text}({data_unit})\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
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

# model_path = "/home/deep/mattersim/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"  # MatterSim 模型路径

# 参数设置
# fmax = 1e20
# name = 'mace'
# calculator = PESCalculator.load_universal(name)

# 初始化 PhononCalc
# phonon_calc = PhononCalc(calculator, t_min=0, t_max=1000, t_step=100, relax_structure=True)

# 加载数据
data_path = "/home/deep/gcnn_keras-master/phonon/phonon_doc.json"
with open(data_path, "r") as f:
    dft_phonon_data = json.load(f)

# 提取材料ID和数据
material_ids = dft_phonon_data["index"]
data = dft_phonon_data["data"]



# 定义模型名称和绘图单位
model_names = [
    "MACE-MPA",
]

# "GRACE-2L-OAM",
#  "MACE-MPA",
#     "SevenNet-MF-ompa",
#     "ORB v3",
#     "DPA-3.1-3M",
#     "MatterSim",

#     "MACE",
#     "SevenNet",
#     "ORB v2",
#     "M3GNet",
#     "CHGNet",
#     "TensorNet"

# 定义6个性质及其单位
properties = {
    "cv_300K": {"unit": "J/K/mol", "title": "Heat Capacity 300K", "name": "Heat capacity"},
    "cv_1000K": {"unit": "J/K/mol", "title": "Heat Capacity 1000K", "name": "Heat capacity"},
    "entropy_300K": {"unit": "J/K/mol", "title": "Entropy 300K", "name": "Entropy"},
    "entropy_1000K": {"unit": "J/K/mol", "title": "Entropy 1000K", "name": "Entropy"},
    "helmholtz_free_energy_300K": {"unit": "kJ/mol", "title": "Free Energy 300K", "name": "Helmholtz free energy"},
    "helmholtz_free_energy_1000K": {"unit": "kJ/mol", "title": "Free Energy 1000K", "name": "Helmholtz free energy"}
}

# 初始化 PhononCalc（每次循环模型时重新初始化）
model_name = 'mace'
def initialize_phonon_calculator(model_name):
    if model_name == "MatterSim":
        # calc = MatterSimCalculator(load_path=model_path, device=device)
        pass
    else:
        calc = PESCalculator.load_universal(model_name.lower())
    return PhononCalc(calc, t_min=0, t_max=1000, t_step=100, relax_structure=False)


# 遍历每个模型
for name in model_names:
    print(f"Processing model: {name}")
    # 初始化每个属性的预测和真实值列表
    predictions_dict = {prop: [] for prop in properties}
    true_labels_dict = {prop: [] for prop in properties}

    # 加载 calculator
    phonon_calc = initialize_phonon_calculator(model_name)

    # 遍历数据，提取结构信息并进行预测
    for idx, entry in tqdm(enumerate(data), total=len(data)):

        try:
            struct_info = entry[0]
            cv_300K = entry[1]
            cv_1000K = entry[2]
            entropy_300K = entry[3]
            entropy_1000K = entry[4]
            helmholtz_free_energy_300K = entry[5]
            helmholtz_free_energy_1000K = entry[6]

            # 将结构信息转换为 pymatgen.Structure 对象
            struct = Structure.from_dict(struct_info)

            # 检查结构是否为有序结构
            if struct.is_ordered:
                try:
                    # 设置温度参数
                    target_temperatures = [300, 1000]
                    result = phonon_calc.calc(struct)

                    # 提取热力学性质
                    thermal_properties = result.get("thermal_properties", {})
                    temperatures = thermal_properties.get("temperatures", [])
                    free_energy = thermal_properties.get("free_energy", [])
                    entropy = thermal_properties.get("entropy", [])
                    heat_capacity = thermal_properties.get("heat_capacity", [])

                    # 提取指定温度下的热力学性质
                    for T in target_temperatures:
                        # 找到最接近目标温度的索引
                        # idx_T = min(range(len(temperatures)), key=lambda i: abs(temperatures[i] - T))

                        idx_T = T // 100
                        # 存储结果
                        predictions_dict[f"cv_{T}K"].append(heat_capacity[idx_T])
                        predictions_dict[f"entropy_{T}K"].append(entropy[idx_T])
                        predictions_dict[f"helmholtz_free_energy_{T}K"].append(free_energy[idx_T] )  # kJ/mol

                    # 存储真实值
                    true_labels_dict["cv_300K"].append(cv_300K)
                    true_labels_dict["cv_1000K"].append(cv_1000K)
                    true_labels_dict["entropy_300K"].append(entropy_300K)
                    true_labels_dict["entropy_1000K"].append(entropy_1000K)
                    true_labels_dict["helmholtz_free_energy_300K"].append(helmholtz_free_energy_300K)
                    true_labels_dict["helmholtz_free_energy_1000K"].append(helmholtz_free_energy_1000K)

                except ValueError as e:
                    # 如果遇到不支持的元素，记录错误并跳过该结构
                    print(f"Skipping structure {struct.composition.reduced_formula}: {e}")
                    continue

        except Exception as e:
            # 捕获 entry 解包时的异常（例如数据格式错误）
            print(f"Error processing entry {idx}: {e}")
            continue

    # 转换为 numpy 数组并过滤 NaN
    for prop in properties:
        y_pred = np.array(predictions_dict[prop])
        y_true = np.array(true_labels_dict[prop])
        valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_pred = y_pred[valid]
        y_true = y_true[valid]

        # 计算 MAE 和 RMSE
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 绘图
        plot_predict_true_high(
            y_pred, y_true,
            data_unit=properties[prop]["unit"],
            model_name=name,
            dataset_name=properties[prop]["title"],
            text=properties[prop]["name"],
            file_path="./"
        )

        # 保存结果
        output_filename = f"{model_name}_{prop}.json"
        with open(output_filename, 'w') as f:
            json.dump({'predictions': y_pred.tolist(), 'true_labels': y_true.tolist()}, f)



