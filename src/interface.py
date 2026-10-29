import torch
import customtkinter as ctk
from customtkinter import filedialog

from utils_interface import center_window
from PIL import Image
from models import transforms, UNetModel
from torchvision import transforms as T

width: int = 1500
height: int = 800
image = None
image_generate = None
preview = None
model = None
MODEL_PATH = "model_unet_best_attempt5.pt"

def load_model() -> None:
    """Создаёт модель один раз и загружает в неё веса"""
    global model
    
    model = UNetModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

def create_window() -> ctk.CTk:
    """Создание главного окна"""
    root = ctk.CTk()
    root.resizable(width = False, height = False)
    root.title("RecoveryNet - рестоврация фото")
    root.geometry(f"{width}x{height}")
    center_window(root, width, height)
    return root

def on_button_load_photo():
    global image
    
    path = filedialog.askopenfilename(
        filetypes=[("Изображения", "*.png *.jpg *.jpeg")]
    )
    
    if not path:
        return
    
    image = Image.open(path).convert("RGB")

def on_button_generate_photo() -> None:
    """Функция для генерации фото"""
    global image_generate
    
    if image is None:
        return
    
    tensor_image = transforms(image).unsqueeze(0)
    
    with torch.no_grad():
        predict = model(tensor_image)
    
    tensor_result = predict.detach().cpu().squeeze(0)
    
    # денормализация: x * std + mean (обратно к Normalize)
    mean = torch.tensor([0.48770615458488464, 0.4506271779537201, 0.4115591049194336]).view(3, 1, 1)
    std = torch.tensor([0.2185600996017456, 0.21286065876483917, 0.21262872219085693]).view(3, 1, 1)
    tensor_result = (tensor_result * std + mean).clamp(0, 1)
    to_pil = T.ToPILImage()
    result_image = to_pil(tensor_result)
    
    image_generate = result_image
        
    # превью в окне
    preview_img = ctk.CTkImage(light_image=image_generate, size=(512, 512))
    preview.configure(image=preview_img)
    preview.image = preview_img

def create_widgets(root: ctk.CTk) -> None:
    """Интерфейс разделён на части: Программа и Окно для вывода результата"""
    global preview
    
    # равная ширина фреймов для программы и для сгенерированого фото
    root.grid_columnconfigure((0,2), weight=1, uniform="half")
    root.grid_rowconfigure(0, weight=1)
    
    # фрейм программа
    program_frame = ctk.CTkFrame(root,  fg_color=("gray88", "gray17"))
    program_frame.grid(row = 0, column = 0, sticky="nsew")
    
    # разделяющая линия
    line = ctk.CTkFrame(root, width=2, fg_color="green")
    line.grid(row=0, column=1, sticky="ns")
    
    # фрейм сгенерированое фото
    generate_frame = ctk.CTkFrame(root, fg_color=("gray92", "gray20"))
    generate_frame.grid(row=0, column=2, sticky="nsew")
    
    
    # текст "программа"
    ctk.CTkLabel(program_frame, text="Программа", font=("Arial", 22)).pack(anchor="n", padx=20, pady=(20, 10))
    
    # Кнопка загрузки фото 
    load_button = ctk.CTkButton(program_frame, text="Загрузить фото",
                                       command=on_button_load_photo)
    load_button.pack(fill="x", padx=20, pady=10)
    
    # Кнопка генерации изображения 
    generate_button = ctk.CTkButton(program_frame, text="Сгенерировать фото",
                                        command=on_button_generate_photo)
    generate_button.pack(fill="x", padx=20, pady=10)
    
    
    # текст "Сгенерированное фото"
    ctk.CTkLabel(generate_frame, text="Сгенерированное фото", font=("Arial", 22)).pack(anchor="n", padx=20, pady=(20, 10))
    
    # само фото (через label!!!)
    preview = ctk.CTkLabel(generate_frame, text="")
    preview.pack(padx=20, pady=10)

def main() -> None:
    load_model()
    root = create_window()
    create_widgets(root)
    
    # Запускаем главный цикл
    root.mainloop()
    
if __name__ == "__main__":
    main()