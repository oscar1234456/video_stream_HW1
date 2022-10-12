import torch
import pandas as pd

from torch.backends import cudnn
from torch.utils.data import DataLoader

from ImageLoader import ImageLoader
from models.model_architecture import VGGNet_19


## for final testing dataset inference
def inference_model(model, testLoader, device):
    all_preds_result = list()

    model.eval()

    with torch.no_grad():
        print(">>> Model Inference Start")
        for batch, inputs in enumerate(testLoader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds_result.extend(preds)

    print(f">> Final Inference End")

    model.train()

    return all_preds_result


if __name__ == "__main__":
    # for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cudnn.deterministic = True

    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load model architecture
    model_ft = VGGNet_19()
    # Load model weight
    model_ft.load_state_dict(torch.load("./final_model_weight/HW1_310551076.pt"))
    model_ft = model_ft.to(device)
    print(model_ft)

    ## DataLoader
    print("Initializing Datasets and Dataloaders...")

    # test path 放測試資料夾根目錄
    test_data = ImageLoader("./data_files/test_pics", 'test', config=None)
    testLoader = DataLoader(test_data, batch_size=32, shuffle=False)

    all_preds_result = inference_model(model_ft, testLoader, device)

    print(all_preds_result)
    test_result = pd.read_csv("./data_files/test_file.csv")
    test_result["label"] =list(map(int, all_preds_result))
    print(test_result)
    test_result.to_csv("./data_files/HW1_310551076.csv", index=False)
    print("csv stored! ->>test completed")
