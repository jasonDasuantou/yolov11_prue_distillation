from ultralytics import YOLO
import os
from utils.yolo.attention import add_attention
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

root = os.getcwd()
## 配置文件路径
name_yaml = os.path.join(root, "data.yaml")
name_pretrain = os.path.join(root, "runs/segment/ori/weights/best.pt")
## 原始训练路径
# path_train = os.path.join(root, "runs/detect/VOC")
name_train = "runs/segment/ori/weights/best.pt"
## 约束训练路径、剪枝模型文件
path_constraint_train = os.path.join(root, "runs/segment/Constraint")
name_prune_before = os.path.join(path_constraint_train, "weights/last.pt")
name_prune_after = os.path.join(path_constraint_train, "weights/prune.pt")
## 微调路径
path_fineturn = os.path.join(root, "runs/detect/VOC_finetune")

def step1_train():
    model = YOLO(name_pretrain)
    model.train(data=name_yaml, device="0", imgsz=720, epochs=50, batch=2, workers=0, save_period=1)  # train the model


## 2024.3.4添加【amp=False】
def step2_Constraint_train():
    model = YOLO(name_train)
    model.train(data=name_yaml, device="0", imgsz=640, epochs=50, batch=2, amp=False, workers=0, save_period=1,
                name=path_constraint_train)  # train the model


def step3_pruning():
    # from utils.yolo.seg_pruning import do_pruning  use for seg
    from utils.yolo.det_pruning import do_pruning  # use for det
    # do_pruning(name_prune_before, name_prune_after)
    do_pruning(name_prune_before, name_prune_after)


def step4_finetune():
    model = YOLO(name_prune_after)  # load a pretrained model (recommended for training)
    for param in model.parameters():
        param.requires_grad = True
    model.train(data=name_yaml, device="0", imgsz=640, epochs=200, batch=2, workers=0, name=path_fineturn)  # train the model

def step5_distillation():
    layers = ["6", "8", "13", "16", "19", "22"]
    model_t = YOLO('runs/segment/ori/weights/best.pt')  # the teacher model
    model_s = YOLO('runs/segment/Constraint/weights/prune.pt')  # the student model
    model_s = add_attention(model_s)
    """
    Attributes:
        Distillation: the distillation model
    """
    model_s.train(data="data.yaml", Distillation=model_t.model, loss_type='mgd',layers=layers, amp=False, imgsz=1280, epochs=300,
                  batch=2, device=0, workers=0, lr0=0.001)


if __name__ == '__main__':
    # step1_train()
    # step2_Constraint_train()
    step3_pruning()
    # step4_finetune()
    # step5_distillation()
