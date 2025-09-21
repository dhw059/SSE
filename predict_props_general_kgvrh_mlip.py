import torch
import numpy as np
import json
from pymatgen.core import Structure
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import json
from pymatgen.core.structure import Structure
import time

from matcalc import PESCalculator, ElasticityCalc
# from mattersim.forcefield import MatterSimCalculator
from tqdm import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True

device = "cuda" if torch.cuda.is_available() else "cpu"


# 参数设置
# fmax = 5.0
opt = "BFGSLineSearch"

# model_path = "/home/deep/mattersim/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
# calculator = MatterSimCalculator(load_path=model_path, device=device)


name = 'chgnet'
calculator = PESCalculator.load_universal(name)

# 加载数据
with open("/home/datasets/all_materials_gvrh_data_processed/all_materials_gvrh_data_processed.json", "r") as f:
# with open("/home/datasets/li_sse_shear_modulus_mp_datasest/li_sse_shear_modulus_mp_datasest.json", "r") as f:
# with open("/home/datasets/include_li_mp_gvrh/include_li_mp_gvrh.json", "r") as f:

# with open("/home/datasets/all_materials_kvrh_data_processed/all_materials_kvrh_data_processed.json", "r") as f:
# with open("/home/datasets/li_sse_bulk_modulus_mp_datasest/li_sse_bulk_modulus_mp_datasest.json", "r") as f:
# with open("/home/datasets/include_li_mp_kvrh/include_li_mp_kvrh.json", "r") as f:
    exp_band_gap_data = json.load(f)


# 提取材料ID和数据
material_ids = exp_band_gap_data["index"]
data = exp_band_gap_data["data"]

# 弹性模量计算
elasticity_calc = ElasticityCalc(calculator, relax_structure=False)

# 初始化预测值和真实值列表
predictions = []
true_labels = []

start_time = time.time()  # Record the start time
# 遍历数据，提取结构信息并进行预测
for entry in tqdm(data, total=len(data)):
    struct_info = entry[0]
    true_shear = entry[1]
    
    if np.isnan(true_shear):
        continue

    # 过滤 shear > 130  bulk > 200 的结构
    # if true_shear <= 130:
    #     continue

    # 将结构信息转换为 pymatgen.Structure 对象
    struct = Structure.from_dict(struct_info)
    # 检查结构是否为有序结构
    if struct.is_ordered:
        try:
            # 尝试进行预测
            elasticity_results = elasticity_calc.calc(struct)
            bulk_modulus_vrh = elasticity_results['shear_modulus_vrh'] * 160.2176
            # bulk_modulus_vrh = elasticity_results['bulk_modulus_vrh'] * 160.2176

            predictions.append(float(bulk_modulus_vrh))
            true_labels.append(true_shear)
        except ValueError as e:
            # 如果遇到不支持的元素，记录错误并跳过该结构
            print(f"Skipping structure {struct.composition.reduced_formula}: {e}")


end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Loop completed in {elapsed_time:.2f} seconds")

# 转换为 numpy 数组
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# 过滤掉包含 NaN 的数据
valid_indices = ~np.isnan(predictions) & ~np.isnan(true_labels)
predictions = predictions[valid_indices]
true_labels = true_labels[valid_indices]


filtered_predictions = predictions
filtered_true_labels = true_labels

# 计算 MAE 和 RMSE
mae = np.mean(np.abs(filtered_predictions - filtered_true_labels))
rmse = np.sqrt(mean_squared_error(filtered_true_labels, filtered_predictions))
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")



# model_name="GRACE-2L-OAM"
# model_name = "MACE-MPA"
# model_name = "SevenNet-MF-ompa"
# model_name = "ORB v3"
# model_name = "DPA-3.1-3M"

# model_name = "MACE"
# model_name = "SevenNet"
# model_name = "ORB v2"
# model_name = "MatterSim"
# model_name = "M3GNet"
model_name = "CHGNet"
# model_name = "TensorNet"


# dataset_name="MPShearModulus"
# dataset_name="MPBulkModulus"

# dataset_name="ContainLiCompounds"
dataset_name="UniversalCompounds"
# dataset_name="Li-SSE Compounds"

data_unit="Gpa"

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
    label_text = f"Shear modulus vrh (Gpa)\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
    ax.text(0.05, 0.95, label_text,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 保存图像
    save_path = os.path.join(file_path, f"{model_name}_{dataset_name}_high.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


# 绘图
plot_predict_true_high_ehull(filtered_predictions, filtered_true_labels)

# 保存结果
output_filename = f"{model_name}_{dataset_name}_high.json"
with open(output_filename, 'w') as f:
    json.dump({'predictions': predictions.tolist(), 'true_labels': true_labels.tolist()}, f)
